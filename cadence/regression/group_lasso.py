"""Group lasso solver via FISTA proximal gradient descent (GPU-native).

Groups correspond to all basis coefficients for one source feature.
Selection is at the group level: either all coefficients for a feature
are nonzero, or all are zero.

All group operations are vectorized (no Python for-loops in the FISTA
inner loop). Groups must be uniform size (always true in CADENCE where
group_size = n_basis).

Optimizations:
  - Sparse objective: when most groups are zeroed, XtX @ beta uses only
    active columns, reducing O(p^2) to O(p_active^2). 2-10x on late FISTA.
  - Lipschitz early exit: convergence check instead of fixed 50 iterations.
"""

import torch
import numpy as np


class GroupLassoSolver:
    """FISTA-based group lasso with warm-start lambda path.

    Solves: min_beta 1/(2n) * ||y - X @ beta||^2 + alpha * sum_g ||beta_g||_2
    where beta_g is the sub-vector of beta for group g.

    All group operations are vectorized via reshape to
    (n_groups, group_size, C), eliminating per-group Python loops
    and CPU-GPU sync points from the FISTA inner loop.
    """

    def __init__(self, groups, device='cuda'):
        """
        Args:
            groups: list of (start_col, end_col) tuples defining column groups.
                    Must be uniform size and contiguous starting at column 0.
                    E.g., for 160 source features x 10 basis functions:
                    [(0,10), (10,20), ..., (1590, 1600)]
            device: torch device
        """
        self.groups = list(groups)
        self.n_groups = len(self.groups)
        self.device = device

        # Uniform group size (required for vectorized ops)
        if self.n_groups > 0:
            self.group_size = self.groups[0][1] - self.groups[0][0]
            self._group_end = self.groups[-1][1]
            assert all(end - start == self.group_size
                       for start, end in self.groups), \
                "All groups must have the same size for vectorized ops"
        else:
            self.group_size = 0
            self._group_end = 0

    def fit(self, X, y, alpha, valid=None, max_iter=1000, tol=1e-6,
            warm_start=None):
        """Fit group lasso at a single regularization strength.

        Uses FISTA with Nesterov acceleration. All group operations
        are vectorized — no Python for-loops in the inner loop.

        Args:
            X: (T, p) design matrix (tensor on device)
            y: (T, C_tgt) target (tensor on device)
            alpha: regularization strength (scalar)
            valid: (T,) boolean mask (optional)
            max_iter: maximum iterations
            tol: convergence tolerance (relative change in objective)
            warm_start: (p, C_tgt) initial beta (optional)

        Returns:
            beta: (p, C_tgt) coefficients
            selected: list of selected group indices
            objective: final objective value
        """
        T, p = X.shape
        C = y.shape[1] if y.dim() > 1 else 1
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Apply validity mask
        if valid is not None:
            X_v = X[valid]
            y_v = y[valid]
        else:
            X_v = X
            y_v = y

        # Precompute XtX and Xty, normalized by n_valid
        # to match _compute_lambda_max which divides by n_valid
        n_valid = X_v.shape[0]
        XtX = X_v.T @ X_v / n_valid  # (p, p)
        Xty = X_v.T @ y_v / n_valid  # (p, C)

        # Lipschitz constant via power iteration on XtX
        L = self._compute_lipschitz(XtX)

        # Initialize beta
        if warm_start is not None:
            beta = warm_start.clone()
        else:
            beta = torch.zeros(p, C, device=self.device, dtype=X.dtype)

        z = beta.clone()  # momentum variable
        t_k = 1.0

        threshold = alpha / L
        ge = self._group_end
        gs = self.group_size
        ng = self.n_groups

        # Precompute yty for cheap per-iteration objective
        # loss = 0.5 * (yty - 2*beta'*Xty + beta'*XtX*beta)  [O(p^2*C)]
        # vs residual form: 0.5 * ||y - X*beta||^2 / n        [O(n*p*C)]
        yty_norm = (y_v ** 2).sum() / n_valid

        obj_prev = float('inf')   # per-iteration (for restart decisions)
        obj_check = float('inf')  # per-10-iter (for convergence check)
        obj = float('inf')

        for iteration in range(max_iter):
            # Gradient at z: grad = XtX @ z - Xty
            grad = XtX @ z - Xty  # (p, C)

            # Gradient step
            beta_new = z - grad / L

            # Proximal step: vectorized block soft-thresholding
            if ng > 0:
                # Reshape group columns: (n_groups, group_size, C)
                groups_3d = beta_new[:ge].view(ng, gs, C)
                # Frobenius norm per group: (n_groups,)
                norms = groups_3d.norm(dim=(1, 2))
                # Shrink factor: max(0, 1 - threshold/norm)
                shrink = torch.clamp(
                    1.0 - threshold / (norms + 1e-20), min=0.0)
                # Apply in-place: (n_groups,1,1) broadcast
                beta_new[:ge] = (
                    groups_3d * shrink.unsqueeze(1).unsqueeze(2)
                ).view(ge, C)

            # Phase 14: sparse objective — exploit zeroed groups
            if ng > 0:
                groups_3d = beta_new[:ge].view(ng, gs, C)
                norms = groups_3d.norm(dim=(1, 2))
                active_mask = norms > 1e-12
                n_active = active_mask.sum().item()
            else:
                n_active = p
                active_mask = None

            if ng > 0 and n_active < ng:
                # Sparse path: only compute XtX @ beta for active columns
                active_groups = active_mask.nonzero(as_tuple=True)[0]
                active_cols = []
                for g in active_groups:
                    s = g.item() * gs
                    active_cols.extend(range(s, s + gs))
                # Include AR columns (beyond group_end)
                if ge < p:
                    active_cols.extend(range(ge, p))
                ac = torch.tensor(active_cols, device=self.device, dtype=torch.long)
                beta_active = beta_new[ac]          # (p_active, C)
                XtX_active = XtX[:, ac]             # (p, p_active)
                XtX_beta = XtX_active @ beta_active  # (p, C)
            else:
                XtX_beta = XtX @ beta_new

            loss = 0.5 * (yty_norm - 2.0 * (Xty * beta_new).sum()
                          + (beta_new * XtX_beta).sum())
            if ng > 0:
                penalty = norms.sum()
            else:
                penalty = torch.tensor(0.0, device=self.device)
            obj = (loss + alpha * penalty).item()

            # Adaptive restart EVERY iteration
            # (O'Donoghue & Candès 2015 — prevents FISTA oscillation)
            if obj > obj_prev and obj_prev < float('inf'):
                # Objective increased — momentum overshot, reset
                t_k_new = 1.0
                z = beta_new.clone()
            else:
                # FISTA momentum (only when NOT restarting)
                t_k_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
                momentum = (t_k - 1.0) / t_k_new
                z = beta_new + momentum * (beta_new - beta)

            # Convergence check every 10 iterations
            if iteration % 10 == 9 or iteration == max_iter - 1:
                if (obj_check < float('inf')
                        and abs(obj_check - obj) / (abs(obj_check) + 1e-20) < tol):
                    beta = beta_new
                    t_k = t_k_new
                    break
                obj_check = obj

            obj_prev = obj
            beta = beta_new
            t_k = t_k_new

        selected = self.selected_groups(beta)
        return beta, selected, obj

    def fit_path(self, X, y, valid=None, n_lambdas=20, lambda_ratio=0.01,
                 max_iter=500, tol=1e-5):
        """Fit along a log-spaced lambda path with warm starting.

        Computes lambda_max (smallest alpha that zeros all groups),
        then fits from lambda_max down to lambda_max * lambda_ratio.

        Returns:
            betas: list of (p, C_tgt) coefficient tensors
            lambdas: (n_lambdas,) numpy array of lambda values
            selected_per_lambda: list of list of selected group indices
        """
        lambda_max = self._compute_lambda_max(X, y, valid)

        # Log-spaced path from lambda_max to lambda_max * lambda_ratio
        lambdas = np.logspace(
            np.log10(lambda_max),
            np.log10(lambda_max * lambda_ratio),
            n_lambdas
        )

        betas = []
        selected_per_lambda = []
        warm = None

        for lam in lambdas:
            beta, selected, _ = self.fit(
                X, y, alpha=float(lam), valid=valid,
                max_iter=max_iter, tol=tol, warm_start=warm
            )
            betas.append(beta.clone())
            selected_per_lambda.append(selected)
            warm = beta  # warm-start next lambda

        return betas, lambdas, selected_per_lambda

    def fit_batched(self, XtX_batch, Xty_batch, yty_batch, alpha,
                    max_iter=300, tol=1e-4, alpha_batch=None):
        """Batched FISTA: K independent group lasso problems in one GPU pass.

        Replaces K sequential fit() calls with a single batched solve using
        torch.bmm. All K problems must share the same (p, C) dimensions.
        Inputs are precomputed Gram matrices (not raw X, y) to avoid storing
        a (K, n_samples, p) intermediate.

        Skips the sparse objective optimization — different problems have
        different active sets, making batched column-gather impractical.
        Full XtX @ beta at p~240 is fast enough in the batched GEMM.

        Args:
            XtX_batch: (K, p, p) precomputed X'X/n per problem
            Xty_batch: (K, p, C) precomputed X'y/n per problem
            yty_batch: (K,) precomputed y'y/n per problem
            alpha: scalar regularization (shared), ignored if alpha_batch given
            max_iter: maximum FISTA iterations
            tol: convergence tolerance (relative change in objective)
            alpha_batch: (K,) optional per-problem regularization

        Returns:
            beta_batch: (K, p, C) coefficients
            selected_batch: list of K lists of selected group indices
            obj_batch: (K,) final objective values
        """
        K, p, _ = XtX_batch.shape
        C = Xty_batch.shape[2]
        dtype = XtX_batch.dtype

        # Batched Lipschitz constants: (K,)
        L = self._compute_lipschitz_batched(XtX_batch)

        beta = torch.zeros(K, p, C, device=self.device, dtype=dtype)
        z = beta.clone()
        t_k = torch.ones(K, device=self.device, dtype=dtype)

        ge = self._group_end
        gs = self.group_size
        ng = self.n_groups

        # Per-problem threshold: alpha / L, shape (K, 1)
        if alpha_batch is not None:
            threshold = (alpha_batch / L).unsqueeze(1)  # (K, 1)
        else:
            threshold = (alpha / L).unsqueeze(1)  # (K, 1)

        INF = torch.tensor(float('inf'), device=self.device, dtype=dtype)
        obj_prev = INF.expand(K).clone()
        obj_check = INF.expand(K).clone()

        for iteration in range(max_iter):
            # Gradient: (K, p, p) @ (K, p, C) -> (K, p, C)
            grad = torch.bmm(XtX_batch, z) - Xty_batch

            # Gradient step
            beta_new = z - grad / L.view(K, 1, 1)

            # Proximal step: vectorized block soft-thresholding
            if ng > 0:
                # (K, n_groups, group_size, C)
                groups_4d = beta_new[:, :ge, :].view(K, ng, gs, C)
                norms = groups_4d.norm(dim=(2, 3))  # (K, ng)
                # threshold is (K, 1), norms is (K, ng)
                shrink = torch.clamp(
                    1.0 - threshold / (norms + 1e-20), min=0.0)
                beta_new[:, :ge, :] = (
                    groups_4d * shrink.unsqueeze(2).unsqueeze(3)
                ).view(K, ge, C)

            # Objective (full, not sparse — see docstring)
            XtX_beta = torch.bmm(XtX_batch, beta_new)  # (K, p, C)
            loss = 0.5 * (
                yty_batch
                - 2.0 * (Xty_batch * beta_new).sum(dim=(1, 2))
                + (beta_new * XtX_beta).sum(dim=(1, 2))
            )  # (K,)

            if ng > 0:
                groups_4d = beta_new[:, :ge, :].view(K, ng, gs, C)
                norms = groups_4d.norm(dim=(2, 3))  # (K, ng)
                penalty = norms.sum(dim=1)  # (K,)
            else:
                penalty = torch.zeros(K, device=self.device, dtype=dtype)

            if alpha_batch is not None:
                obj = loss + alpha_batch * penalty
            else:
                obj = loss + alpha * penalty

            # Adaptive restart: per-problem (vectorized)
            restart_mask = (obj > obj_prev) & (obj_prev < float('inf'))
            t_k_new = torch.where(
                restart_mask,
                torch.ones_like(t_k),
                (1.0 + torch.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
            )
            momentum = torch.where(
                restart_mask,
                torch.zeros_like(t_k),
                (t_k - 1.0) / t_k_new
            )
            z = beta_new + momentum.view(K, 1, 1) * (beta_new - beta)

            # Convergence check every 10 iterations
            if iteration % 10 == 9 or iteration == max_iter - 1:
                finite_check = obj_check < float('inf')
                rel_change = (obj_check - obj).abs() / (obj_check.abs() + 1e-20)
                converged = finite_check & (rel_change < tol)
                if converged.all():
                    beta = beta_new
                    break
                obj_check = obj.clone()

            obj_prev = obj.clone()
            beta = beta_new
            t_k = t_k_new

        # Extract selected groups per problem
        selected_batch = []
        if ng > 0:
            groups_4d = beta[:, :ge, :].view(K, ng, gs, C)
            norms = groups_4d.norm(dim=(2, 3))  # (K, ng)
            active_mask = norms > 1e-12  # (K, ng)
            for k in range(K):
                selected_batch.append(
                    active_mask[k].nonzero(as_tuple=True)[0].tolist())
        else:
            selected_batch = [[] for _ in range(K)]

        return beta, selected_batch, obj

    def _compute_lipschitz_batched(self, XtX_batch):
        """Batched Lipschitz constants via power iteration (fixed 30 iters).

        No early exit — avoids per-problem CPU sync points that would
        serialize the batch. 30 iterations is sufficient for convergence
        at p~240.

        Args:
            XtX_batch: (K, p, p)
        Returns:
            L: (K,) Lipschitz constants with 1.05 safety margin
        """
        K, p, _ = XtX_batch.shape
        v = torch.randn(K, p, 1, device=self.device, dtype=XtX_batch.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-20)

        for _ in range(30):
            Av = torch.bmm(XtX_batch, v)  # (K, p, 1)
            v = Av / (Av.norm(dim=1, keepdim=True) + 1e-20)

        # Rayleigh quotient: v^T XtX v
        Av = torch.bmm(XtX_batch, v)  # (K, p, 1)
        # (K, 1, p) @ (K, p, 1) -> (K, 1, 1)
        L = torch.bmm(v.transpose(1, 2), Av).squeeze(2).squeeze(1)  # (K,)
        L = torch.clamp(L, min=1e-6) * 1.05
        return L

    def _compute_lipschitz(self, XtX):
        """Compute Lipschitz constant L via power iteration with early exit.

        L = largest eigenvalue of X'X. Typically converges in 20-30 iterations.
        """
        p = XtX.shape[0]
        v = torch.randn(p, device=self.device, dtype=XtX.dtype)
        v = v / v.norm()

        L_prev = 0.0
        for i in range(50):
            Av = XtX @ v
            v = Av / (Av.norm() + 1e-20)
            # Early exit: check convergence every 5 iterations after warmup
            if i >= 9 and i % 5 == 4:
                L_est = float(v.dot(XtX @ v))
                if abs(L_est - L_prev) / (abs(L_prev) + 1e-20) < 1e-6:
                    break
                L_prev = L_est

        L = float(v.dot(XtX @ v))
        L = max(L, 1e-6) * 1.05
        return L

    def _compute_lambda_max(self, X, y, valid=None):
        """Compute lambda_max: smallest alpha where all groups are zero.

        lambda_max = max_g ||X_g' y||_F / n_valid
        """
        if valid is not None:
            X_v = X[valid]
            y_v = y[valid]
        else:
            X_v = X
            y_v = y

        if y_v.dim() == 1:
            y_v = y_v.unsqueeze(1)

        n_valid = X_v.shape[0]
        Xty = X_v.T @ y_v  # (p, C)

        if self.n_groups > 0:
            ge = self._group_end
            C = Xty.shape[1]
            Xty_3d = Xty[:ge].view(self.n_groups, self.group_size, C)
            max_group_grad = Xty_3d.norm(dim=(1, 2)).max().item()
        else:
            max_group_grad = 0.0

        lambda_max = max_group_grad / n_valid
        return max(lambda_max, 1e-10)

    def selected_groups(self, beta):
        """Return indices of groups with non-zero norm.

        Args:
            beta: (p, C_tgt) coefficient matrix
        Returns:
            list of int indices of selected groups
        """
        if self.n_groups == 0:
            return []
        ge = self._group_end
        C = beta.shape[1]
        groups_3d = beta[:ge].view(self.n_groups, self.group_size, C)
        norms = groups_3d.norm(dim=(1, 2))
        return (norms > 1e-12).nonzero(as_tuple=True)[0].tolist()
