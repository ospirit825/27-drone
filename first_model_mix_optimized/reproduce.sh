#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Requires TPU-MLIR 1.7. Copy the original first_model.onnx and calib_img/ here.
export OMP_NUM_THREADS=2
model_transform --model_name first_model --model_def first_model.onnx --input_shapes '[[1,3,640,640]]' --mean '0,0,0' --scale '0.003921568627,0.003921568627,0.003921568627' --keep_aspect_ratio --pad_value 114 --pad_type center --pixel_format rgb --test_input calib_img/0000.jpg --test_result first_model_top_outputs.npz --mlir first_model.mlir
for variant in balanced fast; do
  model_deploy --mlir first_model.mlir --quantize INT8 --quantize_table "${variant}_qtable" --chip cv181x --calibration_table baseline_calibration_table --test_input first_model_in_f32.npz --test_reference first_model_top_outputs.npz --model "first_model_sg2002_${variant}_mix.cvimodel"
done
