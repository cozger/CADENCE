"""
Isolate blendshape discrepancy: is it the landmark model or the blendshape model?

Test 3 configurations:
  A) MediaPipe landmarks → MediaPipe blendshapes (baseline)
  B) ONNX landmarks → ONNX blendshapes (our pipeline)
  C) MediaPipe landmarks → ONNX blendshapes (isolates blendshape model)

If C matches A: blendshape ONNX model is correct, issue is in landmark differences
If C differs from A: blendshape ONNX model itself has conversion artifacts
"""
import os
import sys
import time

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

try:
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
except Exception:
    pass

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

BLENDSHAPE_LANDMARK_INDICES = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
    40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
]

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]


def main():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # Init MediaPipe
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(MODELS_DIR, 'face_landmarker.task'))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
    )
    mp_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("[OK] MediaPipe initialized")

    # Init ONNX
    lm_sess = ort.InferenceSession(
        os.path.join(MODELS_DIR, 'face_landmarks_detector.onnx'),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    bs_sess = ort.InferenceSession(
        os.path.join(MODELS_DIR, 'face_blendshapes.onnx'),
        providers=['CPUExecutionProvider'])
    print(f"[OK] ONNX initialized (landmarks: {lm_sess.get_providers()[0]})")

    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Cannot open camera 0")
        sys.exit(1)

    # Warmup
    for _ in range(10):
        cap.read()

    n_frames = 30
    errs_B = []  # ONNX lm → ONNX bs vs MediaPipe bs
    errs_C = []  # MediaPipe lm → ONNX bs vs MediaPipe bs
    timings = []

    print(f"\nRunning {n_frames} frames...\n")

    for i in range(n_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- A: MediaPipe full pipeline ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(frame_rgb))
        result = mp_landmarker.detect(mp_image)

        if not result.face_landmarks:
            continue

        face_lmks = result.face_landmarks[0]
        mp_lm_norm = np.array([[lm.x, lm.y, lm.z] for lm in face_lmks], dtype=np.float32)
        mp_bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)

        # Get face crop from MediaPipe landmarks
        lm_x = mp_lm_norm[:, 0] * w
        lm_y = mp_lm_norm[:, 1] * h
        x_min, x_max = lm_x.min(), lm_x.max()
        y_min, y_max = lm_y.min(), lm_y.max()
        bw, bh = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = max(bw, bh) * 0.625
        c_x1 = max(0, int(cx - half))
        c_y1 = max(0, int(cy - half))
        c_x2 = min(w, int(cx + half))
        c_y2 = min(h, int(cy + half))
        face_crop = frame_rgb[c_y1:c_y2, c_x1:c_x2]
        if face_crop.size == 0:
            continue
        crop_w, crop_h = c_x2 - c_x1, c_y2 - c_y1

        # --- B: ONNX landmarks → ONNX blendshapes ---
        face_256 = cv2.resize(face_crop, (256, 256)).astype(np.float32) / 255.0
        onnx_raw_lm = lm_sess.run(None, {lm_sess.get_inputs()[0].name: face_256[np.newaxis]})[0]
        onnx_raw_lm = onnx_raw_lm.reshape(478, 3)

        subset_B = onnx_raw_lm[BLENDSHAPE_LANDMARK_INDICES, :2]
        onnx_bs_B = bs_sess.run(None, {bs_sess.get_inputs()[0].name: subset_B[np.newaxis]})[0].flatten()

        # --- C: MediaPipe landmarks → ONNX blendshapes ---
        # Convert MediaPipe normalized landmarks to crop-relative pixel coords at 256x256 scale
        mp_lm_crop = mp_lm_norm.copy()
        mp_lm_crop[:, 0] = (mp_lm_norm[:, 0] * w - c_x1) / crop_w * 256.0
        mp_lm_crop[:, 1] = (mp_lm_norm[:, 1] * h - c_y1) / crop_h * 256.0

        subset_C = mp_lm_crop[BLENDSHAPE_LANDMARK_INDICES, :2]
        onnx_bs_C = bs_sess.run(None, {bs_sess.get_inputs()[0].name: subset_C[np.newaxis]})[0].flatten()

        # Compare
        diff_B = np.abs(mp_bs[:52] - onnx_bs_B[:52])
        diff_C = np.abs(mp_bs[:52] - onnx_bs_C[:52])
        errs_B.append(diff_B)
        errs_C.append(diff_C)

        if i < 3 or i % 10 == 0:
            print(f"Frame {i:3d}:")
            print(f"  B (ONNX lm → ONNX bs): mean={diff_B.mean():.4f} max={diff_B.max():.4f}")
            print(f"  C (MP lm   → ONNX bs): mean={diff_C.mean():.4f} max={diff_C.max():.4f}")
            # Show worst 3 for each
            worst_B = np.argsort(diff_B)[-3:][::-1]
            worst_C = np.argsort(diff_C)[-3:][::-1]
            print(f"    B worst: {', '.join(f'{BLENDSHAPE_NAMES[j]}={diff_B[j]:.3f}' for j in worst_B)}")
            print(f"    C worst: {', '.join(f'{BLENDSHAPE_NAMES[j]}={diff_C[j]:.3f}' for j in worst_C)}")

    cap.release()

    if not errs_B:
        print("\nNo valid frames!")
        return

    errs_B = np.array(errs_B)
    errs_C = np.array(errs_C)

    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(errs_B)} frames)")
    print(f"{'='*70}")
    print(f"\nB: ONNX landmarks → ONNX blendshapes  vs  MediaPipe baseline:")
    print(f"   Mean err: {errs_B.mean():.4f}   Max err: {errs_B.max():.4f}")

    print(f"\nC: MediaPipe landmarks → ONNX blendshapes  vs  MediaPipe baseline:")
    print(f"   Mean err: {errs_C.mean():.4f}   Max err: {errs_C.max():.4f}")

    print(f"\n{'='*70}")
    if errs_C.mean() < errs_B.mean() * 0.3:
        print("DIAGNOSIS: Blendshape ONNX model is faithful (C error << B error)")
        print("  The discrepancy comes from LANDMARK coordinate differences,")
        print("  not from the blendshape model conversion.")
        print("  Solution: align face crop before landmark model (like MediaPipe does)")
    elif errs_C.mean() > errs_B.mean() * 0.7:
        print("DIAGNOSIS: Blendshape ONNX model itself introduces error (C ~ B)")
        print("  Even with identical landmarks, the ONNX blendshape model diverges.")
        print("  The TFLite → ONNX conversion may have precision issues.")
    else:
        print("DIAGNOSIS: Both landmark and blendshape models contribute to error")
        print(f"  Landmark contribution: ~{(1 - errs_C.mean()/errs_B.mean())*100:.0f}%")
        print(f"  Blendshape model contribution: ~{errs_C.mean()/errs_B.mean()*100:.0f}%")

    # Per-blendshape breakdown
    print(f"\nPer-blendshape mean error (B vs C):")
    print(f"{'Blendshape':<25s} {'B (ONNX lm)':>12s} {'C (MP lm)':>12s} {'Improvement':>12s}")
    print("-" * 65)
    for j in range(52):
        b_err = errs_B[:, j].mean()
        c_err = errs_C[:, j].mean()
        improvement = (b_err - c_err) / max(b_err, 1e-6) * 100
        if b_err > 0.02:  # Only show significant ones
            print(f"{BLENDSHAPE_NAMES[j]:<25s} {b_err:>12.4f} {c_err:>12.4f} {improvement:>11.1f}%")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
