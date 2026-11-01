# GC4653 装甲板中心坐标检测

代码已部署到荔枝派 `/root/armor_camera/`。模型沿用 `/root/first_model_sg2002_int8_mix.cvimodel`。

## 新增独立摄像头模块

`camera_capture.py` 专门负责启动 GC4653、等待真正的解码画面、获取最新帧和释放资源。主程序已导入这个模块，启动命令保持不变。它不把“端口开放”当成取图成功，也不会关闭其他程序启动的 RTSP 服务。

不启动摄像头，只检查工具、驱动和配置：

```sh
cd /root/armor_camera
python3 camera_capture.py --check
```

接好摄像头并重新上电后，可以先单独拍一张测试图，不运行检测模型：

```sh
python3 camera_capture.py --snapshot camera_test.jpg
```

成功后会保存图片并自动关闭采集；失败时查看 `camera.log`。这比只检查传感器配置更能确认真实取图。

在自己的程序里复用：

```python
from camera_capture import GC4653Camera

with GC4653Camera(log_path="camera.log") as camera:
    sequence, bgr_frame, monotonic_timestamp = camera.read()
    # bgr_frame 是 OpenCV BGR 图片，可传给 preprocess()。
```

本次更新已经在板子通过 5 项摄像头资源/后处理测试、配置检查和静态图片推理回归。没有启动传感器做真实拍摄，摄像头接好后的实际取图仍待验证。

## 已验证与待验证

- 已在当前 SG2002 板子加载模型，直接调用 CVI Runtime，模型仅加载一次。
- 同一输入下，连续三次输出与官方 `model_runner` 完全一致，最大绝对误差 0。
- 已验证静态图片检测、置信度过滤、按类别 NMS、坐标还原和无目标输出。
- 第一张图片 197×177：最高分目标 `1_Blue`，置信度 0.8125，中心 `(90.50,134.21)`。
- 纯推理约 167–169 ms，**不代表摄像头端到端帧率**。
- 当前摄像头未连接：真实取图、排线方向、曝光、实时延迟及稳定性尚未验证。

## 连接摄像头

图片标注为 GC4653；当前系统已有 `libsns_gc4653.so`，`/mnt/data/sensor_cfg.ini` 配置为 `GCORE_GC4653_MIPI_4M_30FPS_10BIT`。

先正常关机、拔掉 USB 电源，再按配套排线和接口标识接入 CSI 摄像头接口，锁好连接器后重新上电。商品图片无法确认排线接点方向，不能据此猜方向或热插拔。

这块板上的 GC4653 走厂商 MMF 管线，不是直接打开 `/dev/video0`。当前实现调用板上官方 `test_mmf 10` 产生本机 RTSP，再由 OpenCV 取图。无需刷机或安装 MaixPy。

## 启动实时检测

登录板子后：

```sh
cd /root/armor_camera
sh run_gc4653.sh
```

只检测蓝色、保存 JSON 和最新标注画面：

```sh
sh run_gc4653.sh --color Blue --save-json results.jsonl --save-image latest.jpg
```

按 Ctrl+C 停止，程序会清理自己启动的摄像头进程和模型。运行前不要同时启动其他占用摄像头的程序。若 MMF 启动失败或没有画面，查看同目录 `camera.log`；此时不能认为摄像头已检测通过。重启板子可清理厂商管线残留。

如已自行启动 RTSP：

```sh
export LD_LIBRARY_PATH=/usr/bin/lib:/mnt/system/usr/lib:$LD_LIBRARY_PATH
python3 armor_detector.py --source rtsp://127.0.0.1:8554/live
```

`test_mmf 10` 当前源码采集 2560×1440 并编码 H265；本机解码会增加 CPU、内存开销和延迟。代码只保留最新解码帧，但不能消除编码器或解码器内部缓存。接上摄像头后需要实测；低延迟版本应进一步改用原生 VPSS 帧直接输入，避免编码再解码。不能把 167 ms 纯推理耗时当作实时画面延迟。

## 输出格式

标准输出每帧一行 JSON，日志走标准错误：

```json
{"frame_id":0,"image_size":[197,177],"found":true,"target":{"class_name":"1_Blue","confidence":0.8125,"center":[90.5,134.21]},"detections":[]}
```

以上为字段节选；实际 `detections` 包含全部保留目标。

- `target`：当前帧最高置信度目标；不进行跨帧跟踪，多目标时可能切换。
- `target.center`：原始输入画面的 `[x,y]` 像素坐标，左上角为 `(0,0)`，右为 X 正向，下为 Y 正向。
- 默认中心为**还原到原画面后的检测框中心** `(x1+x2)/2,(y1+y2)/2`。可用 `--center-method keypoints` 改为四个关键点的算术平均；这不是透视校正后的物理中心。
- `offset_from_image_center`：相对画面中心 `((宽-1)/2,(高-1)/2)` 的像素偏差。
- `center_normalized`：`[x/宽,y/高]`。
- `bbox_xyxy`：边界框；`keypoints`：模型原顺序四个关键点。
- `inference_ms`：模型前向耗时；`frame_age_ms`：从 OpenCV 解码取得该帧到输出前的耗时，不含传感器/编码/网络内部延迟。
- `timestamp`：板子系统时钟；未联网校时可能不是实际日期。
- 无目标时输出 `found:false,target:null,detections:[]`，不复用上一帧位置。

默认置信度阈值 `--conf 0.25`，按类别 NMS 的 IoU 阈值 `--iou 0.45`。输出是二维像素位置，不是距离、三维坐标或云台角度。

## 摄像头没接时测试

```sh
cd /root/armor_camera
export LD_LIBRARY_PATH=/usr/bin/lib:/mnt/system/usr/lib:$LD_LIBRARY_PATH
python3 armor_detector.py --image test_0000.jpg --save-image test_result.jpg
python3 -m unittest test_postprocess
```

源代码中的 `preprocess()`、`postprocess()`、`Runtime.infer()` 分别负责预处理、检测筛选/中心计算和常驻模型推理。

## 参考接口

- [官方 CVI Runtime C API](https://github.com/sophgo/cviruntime/blob/master/include/cviruntime.h)
- [官方 MMF 摄像头与 RTSP 示例](https://github.com/sipeed/LicheeRV-Nano-Build/blob/main/middleware/v2/sample/test_mmf/sample_vio.c)
