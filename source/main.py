import os
import sys
import logging
import uvicorn

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def launch_main_server_start_up(app: FastAPI) -> FastAPI:
    logging.info("Loading Arvane Server components...")
    from .predictor.recon.predictor import ReconPredictor
    from .predictor.depth.predictor import DepthPredictor
    from .analyzer.arvane_analyzer import ArvaneAnalyzer

    from .utils import load_config
    depth_config, recon_config = load_config()
    
    depth_predictor = DepthPredictor(depth_config)
    depth_predictor.init()

    # set app state dependencies
    app.state.depth_predictor = depth_predictor
    app.state.recon_predictor = ReconPredictor(recon_config)

    # load container
    # from .

    return app


from .router.depth import infer_router as infer_depth_router
from .router.depth import update_router as update_depth_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try: 
        app = await launch_main_server_start_up(app)
        yield 
        
    finally: 
        pass

app = FastAPI(lifespan=lifespan)

app.include_router(infer_depth_router)
app.include_router(update_depth_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

uvicorn.run(app, host="127.0.0.1", port=8080)