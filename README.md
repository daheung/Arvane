# Arvane — Monocular 3D Reconstruction Inference Server

**Arvane**은 단안 카메라로부터 전달되는 연속 RGB 프레임, 카메라 내부 파라미터, 카메라 포즈를 누적하여 3차원 공간을 복원하는 GPU 기반 추론 서버입니다.

전체 파이프라인은 다음 세 단계로 구성됩니다.

1. **Monocular Depth Estimation**

   * DepthPro 기반 단안 깊이 추정
2. **Volumetric 3D Reconstruction**

   * TSDF 또는 volumetric representation 기반 프레임 통합
   * RGB, depth, pose, camera intrinsic 및 feature fusion 수행
3. **Semantic Extraction**

   * PointTransformerV3 및 SONATA checkpoint 기반 포인트 특징 추출·분할
   * 분할 결과를 mesh에 반영하고 최종 GLB 생성

서버는 FastAPI 기반 REST API로 구현되며, 재구성 작업은 비동기 task 단위로 관리됩니다. 최종 결과는 웹 및 일반 3D 뷰어에서 사용할 수 있는 **GLB(glTF Binary)** 형식으로 반환됩니다.

---

## 1. System Requirements

### 1.1 Software environment

실행 환경은 다음과 같습니다.

- OS :  Ubuntu 22.04 LTS
- Python : 3.11.13
- CUDA Runtime : 12.4
- PyTorch : 2.8.0+cu180

### 1.2 Hardware requirements

#### Minimum

* Intel Core i5 CPU
* NVIDIA CUDA GPU
* GPU VRAM 16 GiB

#### Recommended

* NVIDIA GPU VRAM 32 GiB
* 다중 모델 로딩과 중간 reconstruction buffer를 수용할 수 있는 시스템 메모리
* CUDA Tensor Core를 지원하는 최신 NVIDIA GPU

Arvane은 단순 free-memory 값이 아니라 추론 중 발생할 추가 메모리 사용량을 고려하여 n개의 디바이스 중 하나를 선택합니다. 따라서 모델이 하나의 디바이스에 바인딩되지 않을 수 있습니다. 최소 조건을 만족하지 않을 시 일부 모델에 대해 Quantization이 수행될 수 있고 GPU가 아닌 CPU에 바인딩 될 수 있습니다.

Quantization에 대한 설정은 depth-dev.yml 파일을 확인하십시오. 해당 기능은 실험적이며 성능에 대한 종속성을 확인중에 있습니다. quantization를 false로 설정하여 비활성화할 수 있습니다.

관련 설정은 다음과 같습니다:
```yaml
quantization: true
quantization_config:
  weight_only: true
  dtype: "int8"
```

Depth 및 Reconstruction 모델은 각각 다음 조건으로 GPU 할당을 요청합니다.

```python
depth_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=6144)
...

recon_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=16384)
...
```

단일 GPU에서 Depth, Reconstruction, Extraction 모델과 입력 데이터, feature volume, TSDF volume 및 중간 tensor를 동시에 유지하려면 **32 GiB 이상의 VRAM을 권장**합니다.

---

## 2. Architecture

Arvane은 다음과 같은 논리 계층으로 구성됩니다.

```text
┌──────────────────────────────────────────────────────────────┐
│                         API Client                           │
│ RGB Frame / Intrinsics / Camera Pose / Task Control         │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Router Layer                      │
│                                                              │
│  /api/world/*       /api/infer/depth       /api/update/*     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       ArvaneEngine                           │
│                                                              │
│  Task lifecycle                                              │
│  Frame accumulation                                          │
│  Pipeline orchestration                                     │
│  Result and status management                               │
└───────────────┬──────────────────┬───────────────────────────┘
                │                  │
                ▼                  ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│   Runtime Layer      │  │         Predictor Layer           │
│                      │  │                                   │
│ TaskStore            │  │ DepthPredictor                    │
│ Executor             │  │ ReconPredictor                    │
│ DeviceManager        │  │ ExtractPredictor                  │
│ Runtime Container    │  │                                   │
└──────────────────────┘  └───────────────┬───────────────────┘
                                          │
                                          ▼
                         ┌────────────────────────────────────┐
                         │ Depth / TSDF / Point Features      │
                         │ Mesh / Semantic Segmentation       │
                         │ GLB Serialization                  │
                         └────────────────────────────────────┘
```

---

## 3. Project Layout

```text
Arvane/
├─ config/
│  ├─ *-dev.yml                    # Development configuration
│  └─ *-prod.yml                   # Production configuration
├─ source/
│  ├─ main.py                      # FastAPI application entry point
│  ├─ engine/                      # ArvaneEngine and pipeline orchestration
│  ├─ router/                      # REST API router definitions
│  ├─ predictor/
│  │  ├─ ...                       # DepthPro predictor
│  │  ├─ ...                       # Volumetric reconstruction predictor
│  │  └─ ...                       # PointTransformerV3/SONATA predictor
│  └─ runtime/                     # Task store, executor, device/container utilities
├─ .env                            # Runtime environment variables
├─ requirements.txt                # Python dependencies
└─ README.md
```

---

## 4. Configuration

Arvane은 `.env`의 `MODE` 환경변수를 기준으로 실행 설정을 선택합니다.

```dotenv
MODE=development
```

설정 선택 규칙은 다음과 같습니다.

| MODE          | Selected configuration |
| ------------- | ---------------------- |
| `development` | `*-dev.yml`            |
| 그 외의 값        | `*-prod.yml`           |

개발 모드에서는 Depth 및 Reconstruction predictor가 각각 개발용 YAML 설정을 로드합니다.

---

## 5. Installation

### 5.1 Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5.2 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3 Start server

```bash
dotenv --file .env run -- python -m source.main
```

---

## 6. Runtime Initialization

애플리케이션 시작 시 `ArvaneEngine`은 다음 predictor를 초기화합니다.

### 6.1 DepthPredictor

DepthPro 모델을 로드하고 단일 RGB 이미지에 대한 metric 또는 relative depth map을 생성합니다.

주요 책임은 다음과 같습니다.

* 이미지 디코딩
* 모델 입력 크기에 맞는 전처리
* GPU tensor 변환
* DepthPro 추론
* depth tensor 후처리
* 출력 dtype 변환

### 6.2 ReconPredictor

누적된 RGB, depth, camera intrinsic 및 camera pose를 사용하여 3차원 공간을 통합합니다.

주요 처리 항목은 다음과 같습니다.

* 입력 프레임 정렬
* RGB/depth 해상도 정규화
* camera intrinsic scaling
* pose 기반 좌표계 변환
* TSDF 또는 volumetric feature 통합
* surface 및 mesh 추출
* GLB 직렬화

### 6.3 ExtractPredictor

PointTransformerV3와 SONATA 계열 checkpoint를 기반으로 reconstruction 결과에 대한 포인트 특징 추출 및 분할을 수행합니다.

주요 처리 항목은 다음과 같습니다.

* mesh 또는 point cloud sampling
* point feature 구성
* PointTransformerV3 inference
* segmentation head 실행
* vertex 또는 face 단위 label 반영
* semantic class 기준 mesh 분리
* ~~색상 또는 material 정보 적용~~

---

## 7. Reconstruction Pipeline

Arvane의 reconstruction 작업은 하나의 `task_id`를 중심으로 수행됩니다.

```text
Create World
    │
    ▼
Accumulate RGB Frames
    │
    ├── Camera Intrinsic K
    ├── Camera Pose
    ├── Timestamp
    └── Optional Depth Estimation
    │
    ▼
Start Reconstruction
    │
    ├── Validate accumulated inputs
    ├── Generate missing depth maps
    ├── Normalize image resolution
    ├── Scale camera intrinsics
    ├── Integrate frames into volume
    ├── Extract mesh
    ├── Run semantic extraction
    ├── Split/color mesh
    └── Serialize final GLB
    │
    ▼
Store Result
    │
    ▼
Return GLB
```

### 7.1 Input normalization

Reconstruction 단계에서 입력 영상은 최종적으로 다음 해상도로 정규화됩니다.

```text
Width  = 640
Height = 480
```

입력 영상 크기가 변경되면 camera intrinsic matrix 역시 동일한 비율로 조정되어야 합니다.

원본 intrinsic matrix가 다음과 같다고 가정합니다.

```text
K = [ fx   0  cx
       0  fy  cy
       0   0   1 ]
```

원본 크기가 `(Wsrc, Hsrc)`, 대상 크기가 `(Wdst, Hdst)`인 경우:

```text
sx = Wdst / Wsrc
sy = Hdst / Hsrc

fx' = fx × sx
fy' = fy × sy
cx' = cx × sx
cy' = cy × sy
```

따라서 resize된 이미지와 원본 intrinsic을 그대로 조합하면 reconstruction 좌표가 왜곡될 수 있습니다.

### 7.2 Pose convention

`pose`는 길이 16의 배열로 전달되는 4×4 transformation matrix입니다.

```text
[ r00 r01 r02 tx
  r10 r11 r12 ty
  r20 r21 r22 tz
   0   0   0   1 ]
```

클라이언트와 서버는 다음 항목에 대해 동일한 convention을 사용해야 합니다.

Pose convention
- Transform: Camera-to-world (T_cw)
- Vector convention: Column vectors
- Transform equation: p_world = T_cw · p_camera
- Coordinate system: Right-handed
- Translation unit: meters

Camera coordinate system
- +X: right
- +Y: down
- +Z: forward

World coordinate system
- +Z: up

Pose convention이 불일치하면 frame alignment 실패, mesh 반전, 축 교환 또는 reconstruction 붕괴가 발생할 수 있습니다.

---

## 8. Task Lifecycle

각 reconstruction 세션은 고유한 `task_id`로 식별됩니다.

개념적인 task 상태는 다음과 같이 구성될 수 있습니다.

```text
CREATED
   │
   ▼
DEPTH
   │
   ▼
RECON
   │
   ▼
EXTRACTING
   │
   ▼
DONE
   │
   ▼
PENDING_KILL
```
작업이 완료된 경우에는 Pending Kill 상태로 전환되며 삭제 대기 상태에 놓이게 됩니다. 또한 오류가 발생한 경우 task는 Aborted 상태로 전환될 수 있습니다.

```text
ABORTED
```

`POST /api/world/start`는 reconstruction을 요청 큐 또는 background executor에 등록한 후 즉시 반환합니다. 따라서 클라이언트는 `status`, `detail`, `result` API를 사용하여 작업 상태를 추적해야 합니다.

---

## 9. REST API

## 9.1 World Lifecycle API

### 9.1.1 Create World

새 reconstruction task를 생성합니다.

```http
POST /api/world/create
Content-Type: application/json
```

Request body:

```json
{
  "user_id": "user123",
  "name": "optional-scene-name"
}
```

Response:

```json
{
  "task_id": "..."
}
```

`task_id`는 이후 update, start, status, detail 및 result 요청에 사용됩니다.

---

### 9.1.2 Update World

RGB frame과 해당 frame의 카메라 정보를 task에 누적합니다.

```http
POST /api/world/update
Content-Type: application/json
```

Request body 예시:

```json
{
  "task_id": "task-id",
  "timestamp": 1720000000,
  "color": {
    "buffer_b64": "data:image/jpeg;base64,..."
  },
  "k_color": [
    525.0, 0.0, 319.5,
    0.0, 525.0, 239.5,
    0.0, 0.0, 1.0
  ],
  "pose": [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
  ],
  "auto_update_depth": true
}
```

#### Fields

| Field               | Type         | Description                       |
| ------------------- | ------------ | --------------------------------- |
| `task_id`           | `string`     | 대상 world task 식별자                 |
| `timestamp`         | `integer`    | 프레임 식별 및 시간순 정렬에 사용하는 키           |
| `color.buffer_b64`  | `string`     | Base64 이미지 또는 Data URL            |
| `k_color`           | `number[9]`  | 3×3 color camera intrinsic matrix |
| `pose`              | `number[16]` | 4×4 camera transformation matrix  |
| `auto_update_depth` | `boolean`    | 프레임 등록 직후 즉시 depth 추론을 실행할지 여부          |

프레임 timestamp는 task 내부에서 중복되지 않는 값으로 관리하는 것이 권장됩니다.

---

### 9.1.3 Start Reconstruction

누적된 입력 데이터를 기반으로 비동기 reconstruction pipeline을 시작합니다.

```http
POST /api/world/start
Content-Type: application/json
```

Request body:

```json
{
  "task_id": "task-id"
}
```

이 API는 GLB 결과가 생성될 때까지 HTTP 연결을 유지하지 않습니다. 요청이 수락되면 background task가 다음 단계를 수행합니다.

1. 입력 데이터 검증
2. 누락된 depth map 생성
3. RGB/depth/intrinsic 전처리
4. volumetric reconstruction
5. mesh 생성
6. semantic extraction
7. mesh split 또는 color assignment
8. GLB 저장

---

### 9.1.4 Get Status

현재 task에 누적된 데이터와 처리 상태를 반환합니다.

```http
GET /api/world/status?task_id=task-id
```

대표적으로 다음 정보가 포함될 수 있습니다.

* 누적 image 개수
* 누적 depth 개수
* camera pose 개수
* intrinsic matrix 개수
* 현재 task 상태
* 결과 생성 여부

Request body 예시:
```
{
    "status": "RECON",
    "num_image": 1180,
    "num_depth": 1180,
    "num_pose": 1180,
    "num_k_image": 1,
    "num_k_depth": 1,
    "reconstruction": {
        "start_init_time": 1784593811.2901587,
        "end_init_time": 1784593811.4037223,
        "num_inits": 1,
        "num_steps": 903,
        "start_final_time": 0,
        "end_final_time": 0,
        "per_view_time": 251.3990194797516
    }
}
```
---

### 9.1.5 Get Detail

Task의 세부 처리 상태와 reconstruction 로그를 반환합니다.

```http
GET /api/world/detail?task_id=task-id
```

상세 정보에는 다음 항목이 포함될 수 있습니다.

* predictor 초기화 상태
* 현재 pipeline 단계
* 단계별 실행 로그
* 처리된 frame 수
* reconstruction step
* 단계별 소요 시간
* 오류 메시지 및 stack context

운영 환경에서는 내부 경로, checkpoint 위치 또는 stack trace가 외부에 직접 노출되지 않도록 응답을 제한해야 합니다.

---

### 9.1.6 Get Result

완성된 reconstruction 결과를 GLB 형식으로 반환합니다.

```http
GET /api/world/result?task_id=task-id
```

#### Processing

결과가 아직 생성되지 않은 경우:

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
```

```json
{
  "status": "processing",
  "message": "The reconstruction result is not ready."
}
```

#### Completed

결과 생성이 완료된 경우:

```http
HTTP/1.1 200 OK
Content-Type: model/gltf-binary
```

Response body에는 GLB binary 데이터가 포함됩니다.

클라이언트는 응답을 UTF-8 문자열이나 JSON으로 변환하지 않고 binary blob으로 처리해야 합니다.

JavaScript 예시:

```javascript
const response = await fetch(
  `/api/world/result?task_id=${encodeURIComponent(taskId)}`
);

if (response.status === 202) {
  const state = await response.json();
  console.log(state);
  return;
}

if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}

const glbBlob = await response.blob();
const objectUrl = URL.createObjectURL(glbBlob);
```

---

## 9.2 Depth Inference API

단일 이미지에 대해 DepthPro 추론만 수행합니다.

```http
POST /api/infer/depth?dtype=float32
Content-Type: image/jpeg
```

Request body에는 인코딩된 이미지 binary를 직접 전달합니다.

```bash
curl \
  -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary "@frame.jpg" \
  "http://localhost:8080/api/infer/depth?dtype=float32" \
  --output depth.bin
```

### Supported dtype

`dtype` query parameter는 다음 값만 허용합니다.

```text
float8
float16
float32
float64
```

실제 `float8` 지원 여부와 저장 표현은 사용 중인 PyTorch 연산 및 NumPy 직렬화 구현에 따라 제한될 수 있습니다.

### Response

```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Depth-Shape: 480,640
Depth-Dtype: float32
Depth-Infer-Time: 0.123
```

Response body에는 별도 container format이 없는 raw depth bytes가 포함됩니다.

클라이언트는 다음 정보를 함께 사용하여 배열을 복원해야 합니다.

* `Depth-Shape`
* `Depth-Dtype`
* byte order
* 배열 저장 순서

Python 예시:

```python
import io

import numpy as np
import requests

with open("frame.jpg", "rb") as image_file:
    response = requests.post(
        "http://localhost:8080/api/infer/depth",
        params={"dtype": "float32"},
        headers={"Content-Type": "image/jpeg"},
        data=image_file.read(),
        timeout=120,
    )

response.raise_for_status()

height, width = map(int, response.headers["Depth-Shape"].split(","))
dtype = np.dtype(response.headers["Depth-Dtype"])

depth = np.frombuffer(response.content, dtype=dtype)
depth = depth.reshape(height, width)

print(depth.shape)
```

---

## 10. Known Limitations

### 10.1 Input resolution

Reconstruction 단계에서 입력 영상은 640×480으로 변환됩니다.

고해상도 입력을 전달하더라도 최종 reconstruction에 사용되는 공간 해상도는 현재 resize 정책의 영향을 받습니다.

### 10.2 GPU memory pressure

다음 요소가 GPU 메모리 사용량을 증가시킵니다.

* DepthPro model parameter
* Reconstruction model parameter
* PointTransformerV3 model parameter
* CUDA allocator cache
* 입력 frame batch
* depth map tensor
* volumetric feature grid
* TSDF volume
* point feature
* mesh extraction 중간 tensor
* `torch.compile` graph 및 compiled kernel cache

단순히 model parameter 크기만 계산해서는 실제 peak VRAM 사용량을 예측하기 어렵습니다.

### 10.3 Task accumulation

World update API를 통해 frame을 계속 누적하면 CPU memory, GPU memory 또는 disk 사용량이 지속해서 증가할 수 있습니다.

운영 환경에서는 다음 정책이 필요합니다.

* task별 최대 frame 수
* 최대 이미지 해상도
* 최대 request body 크기
* task expiration
* 완료 task 정리
* 실패 task 정리
* 동시 reconstruction task 제한
* 사용자별 quota

### 10.4 Pose quality

단안 depth가 정확하더라도 camera pose 오차가 누적되면 reconstruction 품질이 크게 저하될 수 있습니다.

특히 다음 문제가 발생할 수 있습니다.

* surface duplication
* ghost geometry
* texture misalignment
* volume drift
* fragmented mesh
* incorrect scale

Arvane은 depth estimation 서버이면서 동시에 pose-aware reconstruction 서버이므로, 입력 pose의 정확도는 최종 품질에 직접적인 영향을 줍니다. 

해당 문제는 현재 연구 중에 있습니다.

---

## 11. Development Notes

### Run directly

```bash
dotenv --file .env run -- python -m source.main
```

### Check GPU state

```bash
nvidia-smi
```

### Check CUDA from PyTorch

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(
        index,
        properties.name,
        f"{properties.total_memory / 1024**3:.2f} GiB",
    )
PY
```

### Start with development mode

```bash
MODE=development python -m source.main
```

---

## 12. Summary

Arvane은 단순한 단일 이미지 추론 API가 아니라 다음 구성요소를 하나의 task-oriented pipeline으로 결합한 3D reconstruction inference system입니다.

* 단안 RGB 기반 depth estimation
* RGB/depth/pose/intrinsic frame accumulation
* TSDF 또는 volumetric reconstruction
* feature fusion
* PointTransformerV3 기반 point feature extraction
* SONATA checkpoint 기반 semantic segmentation
* mesh split 및 color/material assignment
* GLB binary serialization
* FastAPI 기반 비동기 task lifecycle 관리

최종적으로 Arvane은 연속적인 단안 카메라 입력을 서버 측에서 누적·처리하고, 클라이언트가 직접 렌더링하거나 후속 분석에 사용할 수 있는 GLB 모델을 생성합니다.
