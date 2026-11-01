#!/usr/bin/env python3
"""Convert first_model.onnx to an RV1106 RKNN model using INT8 image calibration.
Run from this directory in a Linux x86_64 environment with a compatible RKNN-Toolkit2.
"""
from pathlib import Path
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
ONNX = ROOT / "first_model.onnx"
DATASET = ROOT / "dataset.txt"
OUT = ROOT / "first_model.rknn"

# Verify calibration list paths and image existence before starting conversion.
image_paths = []
for raw in DATASET.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Calibration image listed in dataset.txt not found: {p}")
    image_paths.append(p)
if not image_paths:
    raise RuntimeError("dataset.txt contains no calibration images")
print(f"ONNX: {ONNX}")
print(f"Calibration images: {len(image_paths)}")
print(f"Output: {OUT}")

rknn = RKNN(verbose=True)
try:
    ret = rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rv1106",
        quantized_algorithm="normal",
        disable_rules=["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"],
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    ret = rknn.load_onnx(model=str(ONNX))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    ret = rknn.build(do_quantization=True, dataset=str(DATASET))
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    ret = rknn.export_rknn(str(OUT))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    if not OUT.is_file() or OUT.stat().st_size == 0:
        raise RuntimeError("export returned success but output file is missing/empty")
    print(f"SUCCESS: {OUT} ({OUT.stat().st_size} bytes)")
finally:
    rknn.release()
