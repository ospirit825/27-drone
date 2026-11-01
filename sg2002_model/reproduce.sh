#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# 在已配置 TPU-MLIR 1.7 的 Python 3.10 / 兼容 Linux 环境中执行。
model_transform --model_name first_model --model_def first_model.onnx --input_shapes '[[1,3,640,640]]' --mean '0,0,0' --scale '0.003921568627,0.003921568627,0.003921568627' --keep_aspect_ratio --pad_value 114 --pad_type center --pixel_format rgb --test_input calib_img/0000.jpg --test_result first_model_top_outputs.npz --mlir first_model.mlir
run_calibration first_model.mlir --dataset calib_img --input_num 50 -o first_model_calibration_table
model_deploy --mlir first_model.mlir --quantize INT8 --quantize_table first_model_qtable --chip cv181x --calibration_table first_model_calibration_table --test_input first_model_in_f32.npz --test_reference first_model_top_outputs.npz --model first_model_sg2002_int8_mix.cvimodel
