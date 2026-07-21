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

- OS : Ubuntu 22.04 LTS
- Python : 3.11.13
- CUDA Runtime : 12.4
- PyTorch : 2.8.0+cu180

### 1.2 Hardware requirements

#### Minimum

* Intel Core i5 CPU
* NVIDIA CUDA GPU
* GPU VRAM 16 GiB

#### Recommendations

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

Depth 및 Reconstruction 모델은 각각 다음 조건으로 GPU 할당을 요청합니다:

```python
depth_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=6144)
...

recon_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=16384)
...
```

단일 GPU에서 Depth, Reconstruction, Extraction 모델과 입력 데이터, feature volume, TSDF volume 및 중간 tensor를 동시에 유지하려면 **32 GiB 이상의 VRAM을 권장**합니다.

---

## 2. Quick Start

### 2.1 Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.2 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Start server

```bash
dotenv --file .env run -- python -m source.main
```

---

## 3. Architecture

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

## 4. Project Layout

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

## 5. Configuration

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

예시 코드는 다음과 같습니다:

```python
TARGET_WIDTH, TARGET_HEIGHT = (640, 480)
_, _, imheight, imwidth = images.shape    # num_image, _, height, width
k_images = k_images[0]
k_images[0] *= TARGET_WIDTH / imwidth
k_images[1] *= TARGET_HEIGHT / imheight
k_images: NDArray = np.array([k_images for _ in range(len(k_image_container))])
```

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

전송 오버헤드 최소화를 위해 color.buffer_b64에 대해 다른 전송 방식을 고려중인 상태입니다.

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

```json
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

---

### 10.5 Pipeline error propagation

Arvane은 서로 독립적으로 학습된 여러 모델과 처리 단계를 직렬 파이프라인으로 결합합니다.

```text
RGB
→ Depth estimation
→ Volumetric/TSDF reconstruction
→ Mesh 또는 point sampling
→ Semantic extraction
→ GLB serialization
```

상위 단계의 출력은 다음 단계의 입력 또는 geometric prior로 사용됩니다. 따라서 각 모델의 오차는 해당 단계에서 끝나지 않고 후속 단계로 전달되며, 일부 오차는 변환·누적 과정에서 증폭될 수 있습니다.

대표적인 오차 전파 경로는 다음과 같습니다.

```text
- depth scale 또는 object boundary 오차가 3D point 위치와 TSDF zero-crossing을 왜곡함
- camera pose 및 intrinsic 오차가 multi-view fusion 과정에서 누적됨
- reconstruction artifact가 extraction 단계의 point sampling과 feature 구성에 영향을 줌
- semantic extraction 모델이 학습 시 보지 못한 분포의 불완전한 geometry를 입력받을 수 있음
- frame별 작은 오차가 시간적으로 누적되어 입력 frame 수가 증가할수록 reconstruction 품질이 저하될 수 있음
- 모델별 해상도, 정규화, 좌표계, 단위 및 confidence calibration 차이가 interface mismatch를 발생시킬 수 있음
```

이 문제는 개별 모델의 독립적인 정확도만으로 전체 파이프라인의 품질을 보장할 수 없다는 구조적 한계입니다. 특정 단계의 성능을 개선하더라도 해당 출력이 후속 모델의 학습 분포와 일치하지 않으면 end-to-end 성능이 동일하게 향상되지 않을 수 있습니다.

현재 공개 구현은 모든 단계를 하나의 최종 목적 함수로 공동 최적화하지 않으며, 상위 단계에서 발생한 오차를 완전히 복구하는 메커니즘도 제공하지 않습니다.

따라서 해당 문제는 현재 연구과제로 남아있습니다.

<details>
  <summary>토글 접기/펼치기</summary> 
  <br/>
  <a href="https://drive.google.com/file/d/1yt9mYvBrJTFX2L_z7tu9P4ygGSuIiBM_/view?usp=drive_link" style="text-decoration: none;">관련 제안서를 보려면 해당 사이트를 참고하세요</a>
</details>

## 11. Troubleshooting

Arvane은 DepthPro, volumetric reconstruction, TSDF/feature fusion, PointTransformerV3 및 SONATA 계열 구현을 하나의 추론 파이프라인으로 통합합니다. 이 과정에서 발생하는 문제는 일반적인 코드 오류뿐 아니라, 각 논문과 공개 구현체가 암묵적으로 사용하는 좌표계, 단위, depth 표현, 카메라 모델, 데이터 전처리 및 수치 정밀도 차이에서 비롯될 수 있습니다.

외부 논문이나 공개 저장소의 코드를 가져올 때는 함수의 입출력 shape만 맞추는 것으로 충분하지 않습니다. 동일한 `pose`, `depth`, `voxel_size`라는 이름을 사용하더라도 구현체마다 실제 의미와 단위가 다를 수 있으므로, 논문의 수식뿐 아니라 공식 코드의 dataset loader, pose preprocessing, intrinsic scaling, depth normalization 및 evaluation 코드를 함께 확인해야 합니다.

### 11.1 외부 논문 및 구현체 통합 점검

통합 전에 다음 항목을 확인하십시오.

```text
- Pose가 camera-to-world인지 world-to-camera인지
- 행렬이 row-vector 또는 column-vector convention을 사용하는지
- 행렬 직렬화가 row-major 또는 column-major인지
- 좌표계가 right-handed 또는 left-handed인지
- 카메라 축의 방향이 무엇인지
- World up axis가 무엇인지
- Translation 단위가 meter, centimeter 또는 millimeter인지
- Depth가 metric depth인지 relative depth인지
- Depth가 camera Z-depth인지 ray distance인지
- Intrinsic이 resize 또는 crop 이전 기준인지
- Voxel size와 TSDF truncation distance의 단위가 무엇인지
```

Pose 행렬은 동일한 4×4 형태라도 의미가 반대일 수 있습니다.

```text
p_world  = T_cw · p_camera
p_camera = T_wc · p_world

T_wc = inverse(T_cw)
```

이를 잘못 해석하면 예외가 발생하지 않더라도 카메라 이동 방향이 반전되거나, 프레임 누적 과정에서 형상이 분리되고 reconstruction이 붕괴할 수 있습니다.

### 11.2 좌표계 및 Pose 검증

Arvane 내부 좌표계 계약은 다음과 같습니다.

```text
Pose matrix shape: 4×4
Pose meaning: camera-to-world
Vector convention: column vector
Transform equation: p_world = T_cw · p_camera
Coordinate system: right-handed
Camera axes: +X right, +Y down, +Z forward
World up axis: +Z
Translation unit: meters
Depth unit: meters
Point coordinate unit: meters
```

`column-major`라는 표현만으로는 행렬의 수학적 의미가 완전히 정의되지 않습니다. 행렬의 메모리 저장 순서와 벡터 곱셈 convention은 별개의 문제이므로 다음 항목을 각각 구분해야 합니다.

```text
1. 행렬의 수학적 의미
2. 벡터를 행렬의 왼쪽 또는 오른쪽에서 곱하는지
3. 메모리 또는 네트워크 직렬화 순서
4. camera-to-world 또는 world-to-camera 여부
```

클라이언트에서 전송한 pose가 서버에서 동일하게 복원되는지 검증하는 것이 권장됩니다.

```python
assert np.allclose(
    received_pose,
    original_pose,
    rtol=1e-6,
    atol=1e-6,
)
```

### 11.3 단위 불일치

Arvane에서는 다음 값이 모두 meter 기준이어야 합니다.

```text
- Depth
- Pose translation
- Camera-space 및 world-space 3D point
- Voxel size
- TSDF truncation distance
- Near/Far distance
- Mesh vertex coordinate
- Extraction radius
- Distance threshold
- Spatial query range
```

예를 들어 depth가 millimeter이고 pose translation이 meter인 경우 다음 값은 숫자상 정상처럼 보입니다.

```text
Depth:             1000
Pose translation:  0.1
```

그러나 실제 의미는 다음과 같습니다.

```text
1000 mm = 1 m
0.1 m   = 100 mm
```

Depth를 meter로 변환하지 않고 사용하면 카메라 이동이 상대적으로 매우 작게 반영되어 여러 프레임이 동일 위치에 중첩될 수 있습니다. 반대로 centimeter 또는 millimeter 단위의 translation을 meter로 해석하면 프레임마다 형상이 수십 배 또는 수천 배 떨어져 생성될 수 있습니다.

모든 단위 변환은 입력 경계에서 한 번만 수행하십시오.

```python
depth_m = depth_mm * 0.001
translation_m = translation_cm * 0.01
```

중간 단계에 임의의 `* 1000`, `/ 1000`, `* 0.01` 연산을 분산시키지 않는 것이 중요합니다. 변수명에도 단위를 포함하는 것이 권장됩니다.

```python
depth_m
translation_m
voxel_size_m
truncation_distance_m
point_camera_m
point_world_m
```

### 11.4 Depth 표현 확인

모든 depth 출력이 meter 단위의 절대 깊이를 의미하는 것은 아닙니다. 외부 depth 모델을 통합할 때는 출력 표현을 확인해야 합니다.

```text
- Metric depth
- Relative depth
- Inverse depth
- Disparity
- Normalized depth
- Camera Z-depth
- Camera-ray distance
```

Inverse depth는 일반적으로 다음 변환이 필요합니다.

```python
depth_m = 1.0 / inverse_depth
```

다만 scale과 shift가 포함된 inverse depth는 단순 역수만으로 metric depth를 얻을 수 없습니다. Relative depth 역시 별도의 scale alignment 없이 metric TSDF에 직접 사용할 수 없습니다.

Camera Z-depth와 ray distance도 서로 다릅니다.

```text
Z-depth:
카메라의 +Z축 방향 거리

Ray distance:
카메라 중심에서 3D 점까지의 유클리드 거리
```

화면 중앙에서는 두 값이 유사하지만 화면 가장자리에서는 차이가 커집니다. 두 표현을 혼동하면 포인트 클라우드가 휘거나 화면 바깥쪽으로 갈수록 표면이 부풀어 보일 수 있습니다.

### 11.5 Resize, Crop 및 Camera Intrinsic

입력 이미지를 resize하거나 crop할 경우 camera intrinsic도 동일한 변환을 적용해야 합니다.

```python
fx_new = fx * scale_x
fy_new = fy * scale_y
cx_new = cx * scale_x - crop_left
cy_new = cy * scale_y - crop_top
```

이미지만 resize하고 intrinsic을 그대로 사용하면 point cloud의 폭과 높이가 왜곡되며, 이 증상은 depth scale 또는 좌표계 오류처럼 보일 수 있습니다.

### 11.6 Voxel 및 TSDF 파라미터

Voxel size와 TSDF truncation distance는 point coordinate와 동일한 단위를 사용해야 합니다.

```python
voxel_size_m = 0.01
truncation_distance_m = 0.04
```

Meter 좌표계에서 위 값은 각각 1 cm와 4 cm를 의미합니다. Point coordinate가 millimeter인데 이를 그대로 사용하면 voxel 크기가 지나치게 작아져 메모리 사용량이 급증하거나, 관측값이 같은 표면으로 fusion되지 않을 수 있습니다.

일반적으로 다음 범위에서 시작할 수 있습니다.

```text
truncation_distance ≈ 3–5 × voxel_size
```

실제 값은 depth noise, 장면 크기 및 reconstruction 방식에 따라 조정해야 합니다.

### 11.7 최적화 이후 정확도 저하

성능 최적화는 기존 구현이 암묵적으로 의존하던 연산 순서, tensor lifetime, precision 및 synchronization을 변경할 수 있습니다.

대표적인 증상은 다음과 같습니다.

```text
- 초기 프레임은 정상이나 누적 후 형상이 붕괴되는 현상
- 동일 입력에서 실행마다 결과가 달라지는 현상
- FP32에서는 정상이나 FP16에서 깨지는 현상
- 단일 요청에서는 정상이나 동시 요청에서 깨재는 현상
- torch.compile 비활성화 시 정상화되는 현상
```

#### Mixed precision

신경망 추론에는 FP16 또는 BF16을 사용할 수 있지만, 기하 연산과 누적 연산은 FP32를 유지하는 것이 안전합니다.

```python
with torch.autocast("cuda", dtype=torch.float16):
    depth_features = depth_model(image)

depth_m = depth_output.float()
pose_c2w = pose_c2w.float()
camera_intrinsic = camera_intrinsic.float()
tsdf_volume = tsdf_volume.float()
```

특히 다음 연산은 FP32 사용을 권장합니다.

```text
- Pose matrix inversion
- Intrinsic matrix inversion
- Camera/world coordinate transformation
- Voxel index calculation
- TSDF weighted accumulation
- Feature weighted accumulation
- Distance 및 threshold 비교
```

#### In-place 연산과 Tensor aliasing

메모리 최적화를 위해 in-place 연산을 사용할 때 동일 storage를 참조하는 tensor가 있는지 확인하십시오.

```python
camera_points = points
world_points = points

world_points.add_(translation)
```

위 코드는 `camera_points`까지 함께 수정합니다. 독립적인 데이터가 필요하면 명시적으로 복사해야 합니다.

```python
world_points = camera_points.clone()
```

Storage 공유 여부는 다음과 같이 확인할 수 있습니다.

```python
print(camera_points.data_ptr())
print(world_points.data_ptr())
```

#### CUDA 비동기 실행

CUDA 연산은 기본적으로 비동기 실행됩니다. 여러 CUDA stream, `non_blocking=True`, pinned memory 또는 FastAPI 동시 요청을 사용할 경우 데이터가 완성되기 전에 다른 단계가 동일 버퍼를 읽거나 덮어쓸 수 있습니다.

문제 위치를 좁히기 위해 개발 환경에서 다음을 사용할 수 있습니다.

```python
torch.cuda.synchronize()
```

```bash
CUDA_LAUNCH_BLOCKING=1
```

해당 설정은 성능을 크게 저하시키므로 production 환경에서는 사용하지 않습니다.

#### Chunk 병렬화와 Race Condition

여러 frame 또는 chunk가 동일 voxel을 동시에 갱신하면 update가 유실될 수 있습니다.

```python
new_tsdf = (
    old_weight * old_tsdf
    + observation_weight * observation_tsdf
) / new_weight
```

다음 항목을 확인하십시오.

```text
- 동일 voxel을 여러 worker가 동시에 수정하는지
- Weight update가 누락되는지
- Feature tensor가 overwrite되는지
- Chunk boundary의 voxel이 중복 처리되는지
- Chunk boundary에 누락된 영역이 존재하는지
- 병렬 처리 순서에 따라 결과가 달라지는지
```

### 11.8 `torch.compile` 및 지연 관련 문제

`torch.compile`은 단순한 실행 속도 향상 기능이 아니라, Python 코드를 실행 그래프로 변환하고 shape, control flow, tensor aliasing 및 side effect를 분석합니다.

다음 요소는 graph break, specialization 또는 buffer 재사용 문제를 유발할 수 있습니다.

```text
- Dynamic shape
- Tensor 값에 의존하는 Python 조건문
- Python list 또는 dictionary mutation
- Global state 변경
- In-place tensor resize
- Custom CUDA operator
- CUDA Graph
- 장기간 보관되는 compiled output tensor
```
#### `DelegateInstExecuter` 클래스를 사용하여 로그 남기기

Arvane에서는 graph break가 감지되는 경우나 특정 시간대에서 지연이 감지되는 경우 이를 남길 수 있는 로그 시스템이 준비되어 있습니다. 

`DelegateInstExecuter` 클래스는 특정 클래스에 대해 인스턴스를 후킹하고 실행 시간을 기록하는 클래스로써 인스턴스의 함수가 실행될 때마다 실행 시간을 측정합니다.

```python
# Create New ReconPredictor due to hooking issue with DelegateInstExecuter. 
# The arvane's predictor instance is not hooked.
predictor = ReconPredictor(recon_config)
predictor.init()

hooker = DelegateInstExecuter(
    ReconPro,
    f"{log_path}/{task_id}.log",
    enable_private_method=True,
)

hooker.set_sink(
    lambda name, dt, ctx: (
        f"[{name}] {dt * 1e3:.2f} ms "
        f"ok={ctx['ok']}\n"
    )
)

hooker.hook_instance(predictor.predictor)
```

이후 해당 클래스에 대해 자원 사용량 변화량을 측정하고 넣는 기능을 추가할 예정입니다.

<details>
<summary>로그 파일 예시</summary> 

```text
...
[get_img_voxel_feats_by_img_bp] 171.47 ms ok=True
[get_img_voxel_feats_by_depth_guided_bp] 177.15 ms ok=True
[predict_per_view] 304.26 ms ok=True
[get_img_voxel_feats_by_img_bp] 174.22 ms ok=True
[get_img_voxel_feats_by_depth_guided_bp] 179.73 ms ok=True
[predict_per_view] 293.63 ms ok=True
[get_img_voxel_feats_by_img_bp] 181.33 ms ok=True
[get_img_voxel_feats_by_depth_guided_bp] 187.88 ms ok=True
[predict_per_view] 315.72 ms ok=True
[get_img_voxel_feats_by_img_bp] 180.05 ms ok=True
[get_img_voxel_feats_by_depth_guided_bp] 187.68 ms ok=True
[predict_per_view] 311.56 ms ok=True
[get_img_voxel_feats_by_img_bp] 172.56 ms ok=True
[get_img_voxel_feats_by_depth_guided_bp] 178.16 ms ok=True
[predict_per_view] 326.29 ms ok=True
...
```
</details>


### 11.9 Reference Mode

최적화된 실행 경로만 유지하면 문제가 좌표계, 단위, precision, concurrency 또는 compile 중 어느 단계에서 발생했는지 판별하기 어렵습니다.

느리더라도 검증 가능한 reference mode를 유지하는 것이 권장됩니다.

```text
Reference Mode
- FP32
- Batch size 1
- 단일 요청
- 단일 CUDA stream
- torch.compile 비활성화
- CUDA Graph 비활성화
- Cache 비활성화
- In-place 연산 최소화
- Chunk 병렬화 비활성화
- Deterministic algorithm 활성화
```

최적화는 다음 순서로 하나씩 활성화합니다.

```text
Reference
→ AMP
→ Batch processing
→ Chunking
→ Async memory copy
→ Concurrent request
→ torch.compile
→ Caching
```

각 단계에서 reference 결과와 비교하고, 문제가 발생한 최초의 최적화 단계에서 원인을 분석하십시오.

### 11.10 합성 데이터 기반 검증

실제 영상은 depth noise, pose noise, texture 부족, motion blur 및 모델 오차가 동시에 포함되므로 원인을 분리하기 어렵습니다.

다음과 같은 합성 입력 테스트를 유지하는 것이 권장됩니다.

#### Identity Pose Test

```text
Depth:              모든 픽셀 1.0 m
Rotation:           Identity
Translation:        (0, 0, 0) m
Expected center:    (0, 0, 1) m
```

#### Translation Test

```text
Depth:              모든 픽셀 1.0 m
Rotation:           Identity
Translation:        (0.1, 0, 0) m
Expected center:    (0.1, 0, 1) m
```

결과가 다음과 같다면 해당 문제를 의심할 수 있습니다.

```text
(100, 0, 1)       Translation 단위 오류
(0.001, 0, 1)     Scale 중복 적용
(0.1, 0, 1000)    Depth millimeter 미변환
(-0.1, 0, 1)      Pose 방향 또는 inverse 오류
(0.1, 0, -1)      Camera Z축 방향 오류
```

Rotation test에서는 방향 벡터에 translation이 적용되지 않도록 homogeneous coordinate의 `w=0`을 사용합니다.

```python
origin_world = T_c2w @ [0, 0, 0, 1]
forward_world = T_c2w @ [0, 0, 1, 0]
right_world = T_c2w @ [1, 0, 0, 0]
```

### 11.11 중간 산출물 저장

전체 reconstruction 결과만 확인하면 어느 단계에서 문제가 발생했는지 판별하기 어렵습니다.

다음 중간 결과를 별도로 저장하는 것이 권장됩니다.

```text
1. Raw depth
2. Converted metric depth
3. Camera-space point cloud
4. World-space point cloud
5. Per-frame pose visualization
6. TSDF volume statistics
7. Extracted mesh
8. GLB export result
```

PLY 등의 중간 산출물을 기준으로 문제를 분리할 수 있습니다.

```text
Camera-space PLY 오류
→ Depth, intrinsic 또는 unprojection 문제

Camera-space PLY 정상 / World-space PLY 오류
→ Pose, 좌표계 또는 단위 문제

World-space PLY 정상 / TSDF 오류
→ Voxel, truncation, 누적 또는 병렬화 문제

TSDF 정상 / GLB 오류
→ Export 축 변환 또는 scale 문제
```

### 11.12 권장 문제 해결 순서

재구성 결과가 비정상적일 때는 다음 순서로 확인하십시오.

```text
1. Depth 표현과 단위 확인
2. Pose 방향과 translation 단위 확인
3. Matrix/vector convention 확인
4. Camera 및 world 축 방향 확인
5. Resize/crop 이후 intrinsic 확인
6. Camera-space point cloud 확인
7. World-space point cloud 확인
8. Voxel size와 truncation 단위 확인
9. FP32 reference mode 실행
10. AMP, chunking, concurrency 및 compile을 하나씩 활성화
11. TSDF와 mesh 결과 비교
12. 마지막으로 GLB export 변환 확인
```

좌표계, 단위 및 최적화 문제는 서로 유사한 증상을 만들 수 있으므로 여러 항목을 동시에 수정하지 않는 것이 중요합니다. 한 번에 하나의 조건만 변경하고 reference output과 비교하십시오.

---

## 13. Summary

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
