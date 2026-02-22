# Arvane — 3D Reconstruction Inference Server (Depth → Recon → Extraction)

Arvane은 **단안(monocular) 입력 프레임 스트림**을 받아서  
1) **Depth 추정(DepthPro)** → 2) **3D Reconstruction(TSDF/volumetric + feature fusion)** → 3) **PointTransformerV3(SONATA) 기반 추출/분할(Extraction)**  
결과를 **GLB(gltf-binary)**로 반환하는 **FastAPI inference 서버**입니다.

---

## 0) Current environment

- Ubuntu 22.04
- CUDA 12.4
- Python 3.11.13
- Torch 2.8.0+cu128

---

## 1) Project layout

```
Arvane/
 ├─ config/                 # depth/recon 설정(yml). MODE에 따라 dev/prod 선택
 ├─ source/
 │   ├─ main.py              # FastAPI 앱 엔트리포인트(uvicorn 0.0.0.0:8080)
 │   ├─ engine/              # ArvaneEngine (depth/recon/extract 파이프라인, task store)
 │   ├─ router/              # REST API 라우터(/api/world, /api/infer/depth, ...)
 │   ├─ predictor/           # DepthPro / Recon / PointTransformerV3(SONATA) predictor
 │   └─ runtime/             # task store, executor, container 등 런타임 유틸
 ├─ .env                     # MODE 등 런타임 환경변수
 ├─ requirements.txt
 └─ README.md
```

- 서버 엔트리: `python -m source.main`
- `.env`의 `MODE`로 config 선택 (`development` → `*-dev.yml`, 그 외 → `*-prod.yml`)

---

## 2) Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env의 MODE를 로드해서 실행
dotenv --file .env run -- python -m source.main
```

- 기본 포트: `0.0.0.0:8080` (uvicorn)
- 개발 편의용 CORS: `allow_origins=["*"]`
- CUDA allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정

---

## 3) Configuration

### 3.1 Depth config (`config/depth-*.yml`)
예시(dev/prod 동일):

- `patch_encoder_preset`, `image_encoder_preset`: `"dinov2l16_384"`
- `checkpoint_uri`: `"source/predictor/checkpoints/depth_pro.pt"`
- `decoder_features`: `256`
- `use_fov_head`: `true`

### 3.2 Recon config (`config/recon-*.yml`)
예시(dev/prod 동일):

- `improved_tsdf_sampling: True`
- `voxel_size: 0.04`
- `depth_guidance.enabled: True`
- `tsdf_fusion_channel: True`
- `checkpoints: source/predictor/checkpoints/arvane.pt`

---

## 4) Runtime model pipeline

### 4.1 엔진 초기화
`ArvaneEngine`이 다음을 초기화합니다.

- `DepthPredictor` (DepthPro)
- `ReconPredictor` (3D recon)
- `ExtractPredictor` (PointTransformerV3 + SONATA head)

또한 `DeviceManager.get_device_considering_slack(required_minimum_memory_mib=16384)`로  
Depth/Recon 각각 **최소 16GB VRAM**을 고려해 디바이스를 바인딩합니다. 권장사양은 **데이터 포함 32GB VRAM** 이 필요합니다.

### 4.2 처리 흐름
`/api/world/start` 호출 시 백그라운드로 아래가 진행됩니다.

1. (옵션/자동) Depth 업데이트  
2. Reconstruction 실행 → GLB/mesh 생성  
3. Extraction(색칠/분할) → mesh split → 최종 결과 저장  

---

## 5) REST API

### 5.1 World lifecycle (Accumulate frame -> Reconstruction -> result)

#### (1) Create world
`POST /api/world/create`

Body:
```json
{
  "user_id": "user123",
  "name": "optional-scene-name"
}
```

Response:
```json
{ "task_id": "..." }
```

#### (2) Update world (Accumulate frame + pose + K)
`POST /api/world/update`

Body 핵심 필드:
- `task_id`: string  
- `timestamp`: int (프레임 키/정렬용)  
- `color.buffer_b64`: base64 인코딩 이미지 (또는 data URL 가능)  
- `k_color`: 길이 9 (3×3 intrinsics)  
- `pose`: 길이 16 (4×4 pose)  
- `auto_update_depth`: (optional) true면 update 후 depth 자동 갱신

#### (3) Start reconstruction ( Asynchronous )
`POST /api/world/start`

Body:
```json
{ "task_id": "..." }
```

#### (4) Status / Detail
- `GET /api/world/status?task_id=...`
  - 누적된 image/depth/pose/K 개수 반환
- `GET /api/world/detail?task_id=...`
  - 재구성 로그(초기화/스텝/시간) 포함

#### (5) Result ( Return GLB )
`GET /api/world/result?task_id=...`

- 준비가 안 됐으면 `202` + 현재 상태 메시지  
- 준비되면 `model/gltf-binary`로 GLB 바이트 반환

---

### 5.2 Depth-only endpoint ( Return depth )
`POST /api/infer/depth?dtype=float32`

- `Content-Type`은 `image/*` 여야 함  
- 응답은 `application/octet-stream` 바디(원시 depth bytes) + 헤더로 shape/dtype 등을 노출

Response headers 예:
- `Depth-Shape: H,W`  
- `Depth-Dtype: float32`  
- `Depth-Infer-Time: ...`

> 주석: `dtype` query는 `float8/16/32/64`만 허용합니다.

---

## 6) Notes / Pitfalls

- **입력 해상도**: recon에서 최종적으로 640×480로 리사이즈 후 처리되며, K도 그에 맞춰 스케일됩니다.
- **VRAM 요구**: depth/recon 각각 16GB 이상을 “슬랙 포함”으로 고려하여 디바이스를 선택합니다.
- **update_depth 라우터**: `/api/update/depth`는 현재 stub(미구현)일 수 있습니다.
- **Extraction 모델**: `facebook/sonata` 체크포인트(seg head 포함)를 로드하고 point transformer 기반으로 추출을 수행합니다.

---
