import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File, UploadFile

from src.containers.containers import Container
from src.routes.routers import router
from src.services.planet_classifier import ProcessPlanet


@router.get("/get_content")
@inject
def get_content(
    config: Container.config = Depends(Provide[Container.config]),
):
    return {
        "code": 200,
        "prediction_names": {name: '' for name in config['model_parameters']['names']},
        "error": None,
    }


@router.post("/process_content")
@inject
def process_content(
    content_image: UploadFile = File(
        ...,
        title="PredictorInputImage",
        alias="image",
        description="Image for inference.",
    ),
    content_process: ProcessPlanet = Depends(Provide[Container.content_process]),
):
    image_data = content_image.file.read()
    content_image.file.close()

    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    dict_process = content_process.process(
        image,
    )
    return {
        "code": 200,
        "predictions": dict_process,
        "error": None,
    }
