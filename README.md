# EV Pothole Detection & GPS-Degraded Visual Localization

## 1. Tổng Quan Bài Toán

Dự án này xây dựng baseline Computer Vision chạy CPU cho hai bài toán trên nền tảng xe điện:

**Phần A - Phát hiện ổ gà và ước lượng độ sâu/diện tích**

Pipeline nhận video từ camera monocular, phát hiện ổ gà bằng YOLOv8s-seg ONNX, ước lượng độ sâu tương đối bằng Depth Anything V2 ONNX, scale về metric bằng giả định ground-plane/camera calibration, sau đó dùng IPM/BEV để tính diện tích bề mặt. Output gồm overlay video, bbox/mask, `area_m2`, `depth_delta_m`, severity và JSONL detection để audit.

**Phần B - Định vị GPS với visual fallback**

Pipeline mô phỏng hệ thống định vị khi GPS tốt/suy giảm/mất tín hiệu. Code đọc KITTI Raw, tạo local XY từ GPS, chạy ORB Visual Odometry, lane detection, GPS integrity monitor, EKF fusion, GPS-loss handover/latch/relock, và landmark database/correction tùy chọn.

Thiết kế chọn monocular camera thay vì stereo để giảm phần cứng và chi phí CPU. Trade-off là depth metric và scale của VO phụ thuộc vào calibration, ground-plane assumption, GPS scale hint hoặc future wheel/IMU/CAN integration.

## 2. Cấu Trúc Thư Mục

```text
.
|-- part_a/                    # Phần A: detect ổ gà, depth/area, benchmark/eval
|-- part_b/                    # Phần B: GPS integrity, VO, lane, EKF, landmark, handover
|-- ros2_nodes/                # ROS 2 replay/deployment interface cho output Part A/B
|-- models/                    # Lưu model ONNX: YOLO pothole, Depth Anything V2, UFLDv2
|-- scripts/                   # Runtime doctor, smoke test, batch demo/artifact runners
|-- tests/                     # Unit test bằng synthetic data
|-- data/                      # Dataset local, calibration YAML, artifact video/JSONL/metrics
|-- README.md                  
|-- report.md                  # Báo cáo kỹ thuật chi tiết và failure analysis
`-- requirements.txt           # Python dependencies
```

## 3. Benchmark Hiện Tại

### 3.1 Part A - Fine-tune YOLO Pothole Segmentation

Model được chọn: `models/yolov8s_pothole.onnx`.

| imgsz | mAP50 Box | mAP50 Mask | Raw ONNX CPU FPS | Ghi chú |
|---:|---:|---:|---:|---|
| 640 | 0.8651 | 0.8543 | 4.6 | Accuracy tốt, quá chậm |
| 512 | 0.8502 | 0.8400 | 7.1 | Vẫn chậm |
| 480 | 0.8462 | 0.8301 | 8.1 | Vẫn chậm |
| 448 | 0.8266 | 0.8123 | 9.3 | pass mAP >= 0.80 và cân bằng FPS/accuracy |
| 416 | 0.7982 | 0.7834 | 11.3 | Fail ngưỡng mAP box/mask 0.80 |
| 320 | 0.6402 | 0.6263 | 19.0 | Chỉ dùng latency stress test |
| 288 | 0.5696 | 0.5588 | 21.6 | Không dùng final |

### 3.2 Part A - Full Pipeline FPS

Artifact: `data/phase2b_outputs/benchmark_448_render300.json`.

| Metric | Value |
|---|---:|
| Số frame | 300 |
| Avg end-to-end FPS | 11.24 |
| p50 / p95 latency | 896.08 / 1046.56 ms |
| Avg YOLO detect | 85.68 ms |
| Avg depth effective | 637.95 ms |
| Peak memory | 1857.05 MB |

Kết quả này chưa đạt KPI 15 FPS vì Depth Anything V2 là bottleneck lớn nhất trên CPU.

### 3.3 Part A - Demo Trên 5 Video Thật

Artifact: `data/phase2b_outputs/real_demos/manifest.json`.

| Video | Frames | Detection records | Avg FPS | p50 / p95 latency | Avg YOLO | Avg depth effective |
|---|---:|---:|---:|---:|---:|---:|
| vid1.mp4 | 1160 | 212 | 10.95 | 82.51 / 261.35 ms | 79.07 ms | 28.47 ms |
| vid2.mp4 | 419 | 138 | 9.89 | 82.89 / 834.70 ms | 80.41 ms | 53.60 ms |
| vid3.mp4 | 1805 | 629 | 10.01 | 84.94 / 810.61 ms | 80.68 ms | 50.11 ms |
| vid4.mp4 | 1294 | 448 | 9.92 | 85.37 / 845.08 ms | 81.85 ms | 50.10 ms |
| vid5.mp4 | 3680 | 1578 | 9.66 | 85.13 / 823.93 ms | 80.71 ms | 56.78 ms |

Tổng batch real-demo có 8358 frames và 3005 detection records. FPS tốt hơn benchmark dense vì depth chỉ chạy khi có detection và có cache theo `--depth-every-n`.

### 3.4 Part A - Depth/Area GT Audit

Artifact: `data/phase2b_outputs/real_demos/vid1_depth_area_eval.json`.

| Metric | Value |
|---|---:|
| GT rows / matched rows | 212 / 212 |
| Area evaluated samples | 26 |
| Area MAE | 0.01168 m2 |
| Area median abs error | 0.00715 m2 |
| Area MAPE / median % / p95 % | 8.48% / 4.89% / 18.77% |
| Depth_delta evaluated samples | 26 |
| Depth_delta MAE | 0.00383 m |
| Depth_delta median abs error | 0.00057 m |
| Depth_delta MAPE / median % / p95 % | 16.08% / 6.77% / 47.81% |

Area khá ổn trong audit nhỏ, còn depth có median tốt nhưng bị kéo bởi một số outlier. Ground truth area hiện là ellipse approximation từ đo tay và tính toán, chưa phải diện tích biên dạng vật lý tuyệt đối.

### 3.5 Part B - GPS-Good VO/Lane/EKF Baseline

Artifacts: `data/phase3_outputs/ufldv2_0001_clean.metrics.json` và `.summary.json`.

| Metric | Value |
|---|---:|
| Frames / duration | 108 / 11.04 s |
| VO valid frames | 107 / 108 (99.07%) |
| Avg VO matches / inliers | 950.55 / 403.20 |
| GPS / VO / fused path length | 102.44 / 102.34 / 104.04 m |
| VO error vs GPS mean / p95 / max | 55.64 / 94.43 / 96.87 m |
| Final VO error / final EKF fused error | 96.87 / 0.087 m |
| Estimated processing FPS | 3.19 |
| Avg VO / lane / fusion-event time | 56.17 / 256.91 / 0.263 ms |

Lane-side manual sanity audit đạt `22/22` đúng cho class `right`. Đây chưa phải full accuracy vì chưa kiếm được các sequence có nhãn class `left`.

### 3.6 Part B - GPS-Loss Handover

Artifact: `data/phase3_outputs/phase5_handover_latest.metrics.json`.

| Metric | Value |
|---|---:|
| Frames | 80 |
| GPS states | 44 good / 4 degraded / 32 lost |
| Transitions | good->degraded, degraded->lost, lost->good |
| Visual fallback error mean / p95 / max | 0.98 / 1.21 / 1.22 m |
| Relock error | 1.21 m |
| VO valid ratio | 98.75% |
| Estimated processing FPS | 20.52 |
| VO scale source | 43 gps_good / 37 last_good_gps_scale / 0 unit_fallback |

### 3.7 Part B - Landmark Smoke Test

Artifacts: `data/phase4_landmarks/smoke_0001.landmarks.jsonl.summary.json` và `data/phase4_landmarks/smoke_0001.landmark_eval_latest.json`.

| Metric | Value |
|---|---:|
| Frames | 30 |
| Observations | 382 |
| Unique landmarks | 14 |
| New landmarks | 14 |
| Proxy re-identification rate | 96.34% |
| Street-name sign proxy rate | 95.39% |
| Traffic sign proxy rate | 96.96% |

Đây là proxy association, không phải GT recall/precision.

## 4. Command Chạy Demo Và Test

Trong các command dưới đây, có thể dùng `python` nếu môi trường đã setup, hoặc thay bằng virtual env trên Windows.

### 4.1 Setup Và Smoke Check

Setup môi trường ảo (virtual env)
```powershell
python -m venv myenv
pip install --upgrade pip
pip install -r requirements.txt
```
Run test
```powershell
py -B scripts\runtime_doctor.py
py -B scripts\smoketest_p0.py
```

### 4.2 Unit Test

Unit test không cần video thật hoặc model ONNX.

```powershell
python -B -m compileall -q part_a part_b ros2_nodes scripts tests
python -m unittest discover -s tests
```

### 4.3 Part A - Chạy Demo Một Video

```powershell
python -B -m part_a.pipeline `
  --source data/demo_inputs/vid1.mp4 `
  --calib data/calibration/vid1_1080_1112_assumed.yaml `
  --output data/phase2b_outputs/vid1_pipeline_demo.mp4 `
  --imgsz 448 `
  --depth-every-n 4
```

### 4.4 Part A - Benchmark

```powershell
python -B -m part_a.benchmark `
  --source data/demo_inputs/mendeley_pothole_test_all.mp4 `
  --calib data/calibration/mendeley_1080_assumed.yaml `
  --summary data/phase2b_outputs/benchmark_448_render300.json `
  --detections data/phase2b_outputs/benchmark_448_render300_detections.jsonl `
  --output data/phase2b_outputs/part_a_demo_448_render300.mp4 `
  --max-frames 300 `
  --process-all-frames `
  --imgsz 448 `
  --depth-every-n 4
```

### 4.5 Part A - Batch 5 Video Thật

```powershell
python -B scripts/run_part_a_real_demos.py `
  --output-dir data/phase2b_outputs/real_demos
```

### 4.6 Part A - Eval Depth/Area

```powershell
python -B -m part_a.depth_area_eval `
  --detections data/phase2b_outputs/real_demos/vid1_detections.jsonl `
  --ground-truth data/phase2b_outputs/real_demos/vid1_depth_area_gt.csv `
  --metrics data/phase2b_outputs/real_demos/vid1_depth_area_eval.json `
  --matches data/phase2b_outputs/real_demos/vid1_matches_audit.jsonl
```

### 4.7 Part B - KITTI VO/Lane/EKF Baseline

```powershell
python -B -m part_b.pipeline `
  --data-root data `
  --sequence 0001 `
  --lane-backend ufldv2 `
  --lane-side-mode binary_road `
  --lane-every-n 3 `
  --output data/phase3_outputs/ufldv2_0001_binary_lane.jsonl

python -B -m part_b.metrics `
  --input data/phase3_outputs/ufldv2_0001_binary_lane.jsonl `
  --metrics data/phase3_outputs/ufldv2_0001_binary_lane.metrics.json `
  --plot-dir data/phase3_outputs/plots

python -B -m part_b.render `
  --sync data/2011_09_26_drive_0001_sync.zip `
  --phase3-output data/phase3_outputs/ufldv2_0001_binary_lane.jsonl `
  --calib data/2011_09_26_drive_0001_calib.zip `
  --output data/phase3_outputs/ufldv2_0001_binary_lane_overlay.mp4
```

### 4.8 Part B - GPS-Loss Handover

```powershell
python -B scripts/run_part_b_handover_artifact.py `
  --data-root data `
  --sequence 0001 `
  --lane-backend heuristic `
  --lane-side-mode binary_road `
  --max-frames 80 `
  --gps-loss-start 20 `
  --gps-loss-end 55 `
  --gps-loss-degraded-frames 4 `
  --output data/phase3_outputs/phase5_handover_latest.jsonl `
  --metrics data/phase3_outputs/phase5_handover_latest.metrics.json
```

### 4.9 Part B - Landmark DB Smoke Test

```powershell
python -B -m part_b.build_landmarks `
  --data-root data `
  --sequence 0001 `
  --max-frames 30 `
  --output data/phase4_landmarks/smoke_0001.landmarks.jsonl

python -B -m part_b.landmark_eval `
  --observations data/phase4_landmarks/smoke_0001.landmarks.jsonl `
  --metrics data/phase4_landmarks/smoke_0001.landmark_eval_latest.json
```

### 4.10 ROS 2 Replay

```powershell
ros2 launch ev_pothole_localization_nodes replay_demo.launch.py `
  detections_jsonl:=data/phase2b_outputs/real_demos/vid1_detections.jsonl `
  image_source:=data/demo_inputs/vid1.mp4 `
  phase3_jsonl:=data/phase3_outputs/phase5_handover_latest.jsonl
```

## 5. Limitations

- Dự án hiện là demo/baseline kỹ thuật, chưa phải production-ready vehicle stack.
- Part A full pipeline CPU chưa đạt KPI 15 FPS vì Depth Anything V2 là bottleneck chính.
- Calibration YAML hiện là assumed, chưa phải checkerboard calibration đo thật; depth/area metric phụ thuộc mạnh vào camera height, pitch và mặt đường phẳng.
- GT depth/area còn nhỏ: `vid1` có 26 mẫu thủ công được eval; GT area dùng ellipse approximation.
- Chưa validate đầy đủ robustness ban đêm/mưa/nắng gắt bằng nhãn điều kiện thực tế.
- Monocular VO drift lớn nếu chạy dài hạn mà không có loop closure, bundle adjustment, VIO hoặc wheel/CAN scale.
- GPS-loss handover dùng KITTI + simulator, chưa phải dữ liệu hầm/bãi xe thật tại Việt Nam.
- Lane-side audit hiện chỉ cover class `right`; chưa claim full left/right accuracy.
- Landmark evaluation mới là proxy association; chưa có GT `observation_id -> gt_landmark_id`, nên chưa claim recall/precision.
- ROS 2 nodes hiện là replay/deployment-interface baseline, dùng JSON `std_msgs/String` cho pothole thay vì custom message/lifecycle/QoS production.

