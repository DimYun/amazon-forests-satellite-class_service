import uvicorn
import os
import cv2
from PIL import Image
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, APIRouter, Depends, UploadFile, File

from container_task import Container, ProcessPlanet, Storage

router = APIRouter()


@router.get('/get_content')
@inject
def get_content(
    content_id: str,
    storage: Storage = Depends(Provide[Container.store])
):
    # print(storage.get(content_id))
    return {
        'content': storage.get(content_id),
    }


@router.post('/process_content')
@inject
def process_content(
    content_image: UploadFile = File(...),
    content_process: ProcessPlanet = Depends(Provide[Container.content_process]),
):
    try:
        contents = content_image.file.read()
        with open(os.path.join('uploaded_imgs', content_image.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        content_image.file.close()

    image = cv2.imread('test.jpg')[..., ::-1]
    Image.fromarray(image)
    str_process = content_process.process(
        image,
        str(content_image.filename).split('.')[0],
    )
    return {
        "message": f"Successfully uploaded {content_image.filename}",
        "scores": str_process
    }


def create_app():
    config = {'content_process': {'dir_path': 'api_test_dir'}}
    container = Container()
    container.config.from_dict(config)
    container.wire([__name__])
    app = FastAPI()
    app.container = container
    app.include_router(router)
    # app.container = container
    # app.include_router(endpoints.router)
    return app


if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, port=2134)
