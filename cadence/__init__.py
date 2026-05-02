"""CADENCE — Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation."""

# Preload torch before any numpy-dependent submodule imports it.
# Windows-specific: torch 2.10 + numpy 2.4 has a DLL-load ordering bug where
# numpy's MKL installs a DLL search hook that breaks torch's later shm.dll load.
# Loky worker subprocesses re-enter this package via cloudpickle and hit the
# same issue if numpy gets loaded first, so we hoist torch at the package root.
import torch as _torch  # noqa: F401

__version__ = '0.1.0'
