import os
import sys
import json
import logging
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

MAX_LOG_BODY = 2048  # 바디 로그는 2KB만 저장(과도한 로그/개인정보 보호)

def _format_validation_errors(errors: list[dict]) -> list[str]:
    out = []
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", []))   # 예: body.rotation.3
        msg = e.get("msg")
        typ = e.get("type")
        out.append(f"{loc}: {msg} ({typ})")
    return out

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def launch_main_server_start_up(app: FastAPI) -> FastAPI:
    # from .predictor.recon.predictor import ReconPredictor
    # from .predictor.depth.predictor import DepthPredictor
    # from .analyzer.arvane_analyzer import ArvaneAnalyzer

    # from .utils import load_config
    # depth_config, recon_config = load_config()

    logging.info("Loading Arvane Server Engine ...")
    from .engine.arvane_engine import ArvaneEngine
    app.state.engine = ArvaneEngine()
    
    # depth_predictor = DepthPredictor(depth_config)
    # depth_predictor.init()

    # recon_predictor = ReconPredictor(recon_config)
    # recon_predictor.init()

    # # set app state dependencies
    # app.state.depth_predictor = depth_predictor
    # app.state.recon_predictor = recon_predictor

    # # load container
    # from .runtime.infer.store import InferenceChunkStoreConcurrent
    # app.state.container = InferenceChunkStoreConcurrent(chunk_size=64)

    return app


from .router.depth import infer_router as infer_depth_router
from .router.depth import update_router as update_depth_router
from .router.image import image_router as image_router
from .router.world.world import world_router as world_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try: 
        app = await launch_main_server_start_up(app)
        yield 
        
    finally: 
        pass

app = FastAPI(lifespan=lifespan)

app.debug = True

app.include_router(infer_depth_router)
app.include_router(update_depth_router)
app.include_router(image_router)
app.include_router(world_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        raw = await request.body()
        body_text = raw.decode("utf-8", errors="replace")
    except Exception:
        body_text = "<failed-to-read-body>"
    if len(body_text) > MAX_LOG_BODY:
        body_text = body_text[:MAX_LOG_BODY] + f"... <truncated {len(body_text)-MAX_LOG_BODY} bytes>"

    formatted = _format_validation_errors(exc.errors())
    logging.error("422 ValidationError: %s %s", request.method, request.url.path)
    logging.error("errors=%s | body=%s", formatted, body_text)

    # 클라이언트에도 어떤 필드가 문제인지 돌려주고 싶다면:
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),         # pydantic v2 표준 포맷
            "invalid_fields": formatted,    # 사람이 보기 쉬운 요약
        },
    )
    
uvicorn.run(app, host="0.0.0.0", port=8080)
