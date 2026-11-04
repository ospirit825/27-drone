# first_model：荔枝派 RV Nano 混合精度候选模型

本包为2026-11-04新编译的两个候选版本。使用用户 `Rm-Oct-1` 文件夹内随 `first_model.pt` 提供的 `first_model.onnx`，没有重新训练。源文件SHA-256见 `model_manifest.json`。ONNX与之前已部署模型的导出来源相同；本次变化在混合精度编译表。

## 文件选择

| 模型 | 显式BF16表项数 | 文件大小 | 工具报告ION需求 | 建议 |
|---|---:|---:|---:|---|
| first_model_sg2002_balanced_mix.cvimodel | 53 | 3,149,144字节 | 12.43 MB | 推荐先试，保留注意力和坐标解码的敏感表项 |
| first_model_sg2002_fast_mix.cvimodel | 27 | 3,140,112字节 | 12.37 MB | 速度优先的实验候选，进一步缩小BF16保留范围 |

对照旧版本有77个BF16表项，文件3,400,104字节，ION需求13.20 MB。表项数量不等于实际BF16算子数量或耗时，部分操作可能被融合或由编译器自动指定精度。

**两版都未连接真实开发板测速，不能保证比旧版快，也没有验证26～30 FPS。** “fast”表示编译策略目标，不是速度已达标。文件变小、内存减少不直接等于相同幅度的推理加速。

## 兼容接口

- SG2002，编译目标 `cv181x`，cvimodel格式1.4.0。
- 输入：`images`，FP32，`[1,3,640,640]`，RGB/NCHW。
- 先等比例缩放，尺寸使用 `int(w*r),int(h*r)`，居中补114，再除255。模型未融合图像预处理。
- 输出：`output0_Concat_f32`，FP32，`[1,36,8400,1]`。
- 36通道仍为4个xywh、24类分数和4个关键点的8个坐标；没有独立objectness。仍需原来的NMS和坐标还原。
- 类别顺序见 `model_config.json`。输入输出尺寸与现有 `armor_detector.py` 相同。

## 验证结果与限制

1. TPU-MLIR 1.7浮点变换与ONNX比较，317个张量全部通过。
2. 两个候选均通过模型部署工具的混合量化输出比较、编译模型与TPU中间输出比较。
3. 另用官方CV181x模拟器比较8张图片：4张校准图片、4张先前视频帧，参考为ONNX浮点输出。它们不是人工标注测试集，不用于宣称mAP或真实精度。
4. 旧版、balanced均在8张图有检测输出，累计10个候选；fast在7张图有输出，累计9个候选。fast在 `video_00043310.jpg` 未保留低置信度目标，而旧版和balanced保留了它。不能把这些数字当成准确率。
5. 模拟输出存在分数和坐标的小幅变化，不能声称与旧模型逐元素一致。详细结果见 `validation/candidate_evaluation.json`。
6. 全输出余弦相似度容易被坐标通道主导，不足以单独证明检测效果；因此同时保存了检测框、类别和中心对比。

## 使用

暂时保留旧模型，不要覆盖。将balanced文件上传到板子 `/root/` 后，在已有程序目录执行：

```sh
cd /root/armor_camera
export LD_LIBRARY_PATH=/usr/bin/lib:/mnt/system/usr/lib:$LD_LIBRARY_PATH
python3 armor_detector.py --model /root/first_model_sg2002_balanced_mix.cvimodel --image test_0000.jpg --save-image balanced_result.jpg
```

测试fast时只替换 `--model` 指向的文件。摄像头已接好时，可沿用 `sh run_gc4653.sh --model /root/first_model_sg2002_balanced_mix.cvimodel`。

后续板端评估需使用相同输入、预热、线程和计时范围，分别比较前向与完整流程耗时，再按视频检测效果决定是否替换。模拟器运行时间不是板端速度。

## 复现

工具链：官方 `tpu_mlir-1.7-py3-none-any.whl`，版本v1.7-20240419。

保留了本次编译精度表、复用的50张图片校准表和 `reproduce.sh`。复现需自行放入原始 `first_model.onnx` 和 `calib_img/`；本包不重复包含权重来源和校准图片。

参考：[TPU-MLIR混合精度说明](https://doc.sophgo.com/sdk-docs/v23.09.01-lts-sp5/docs_latest_release/docs/tpu-mlir/quick_start_en/html/07_quantization.html)。
