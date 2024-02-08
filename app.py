import argparse

import uvicorn
from omegaconf import OmegaConf

from src.containers.containers import Container
from src.routes.routers import router as app_router
from src.routes import planets as planets_routes

from fastapi import FastAPI


def create_app() -> FastAPI:
    container = Container()
    cfg = OmegaConf.load('configs/config.yaml')
    container.config.from_dict(cfg)
    container.wire([planets_routes])

    app = FastAPI()
    app.include_router(app_router, prefix='/planets', tags=['planet'])
    return app


if __name__ == "__main__":

    def arg_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("port", type=int, help="port number")
        return parser.parse_args()

    app = create_app()
    args = arg_parse()
    uvicorn.run(app, port=args.port, host="127.0.0.1")
