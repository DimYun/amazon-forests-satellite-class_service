import numpy as np
import cv2
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File, UploadFile
from PIL import Image

from src.routes.routers import router
from src.containers.containers import Container
from src.services.planet_classifier import ProcessPlanet, Storage


@router.get("/get_content")
@inject
def get_content(
    content_id: str,
    storage: Storage = Depends(Provide[Container.store]),
):
    return {
        "content": storage.get(content_id),
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
    try:
        image_data = content_image.file.read()
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        content_image.file.close()

    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    Image.fromarray(image)
    str_process = content_process.process(
        image,
        str(content_image.filename).split(".")[0],
    )
    return {
        "message": f"Successfully uploaded {content_image.filename}",
        "scores": str_process,
    }
