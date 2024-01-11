import uvicorn
import os
import cv2
from PIL import Image
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, APIRouter, Depends, UploadFile, File

from src.container_task import Container, ProcessPlanet, Storage

router = APIRouter()


@router.get('/get_content')
@inject
def get_content(
    content_id: str,
    storage: Storage = Depends(Provide[Container.store]),
):
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
        with open(os.path.join('uploaded_imgs', content_image.filename), 'wb') as f:
            f.write(content_image.file.read())
    except Exception:
        return {'message': 'There was an error uploading the file'}
    finally:
        content_image.file.close()

    image = cv2.imread(os.path.join('uploaded_imgs', content_image.filename))[..., ::-1]
    Image.fromarray(image)
    str_process = content_process.process(
        image,
        str(content_image.filename).split('.')[0],
    )
    return {
        'message': f'Successfully uploaded {content_image.filename}',
        'scores': str_process,
    }


def create_app():
    config = {
        'content_process': {
            'dir_path': 'api_test_dir',
            'dir_upload': 'uploaded_imgs',
        },
    }
    container = Container()
    container.config.from_dict(config)
    container.wire([__name__])
    app = FastAPI()
    app.container = container
    app.include_router(router)
    return app


def set_routers(app: FastAPI):
    app.include_router(router, prefix='/planet', tags=['planet'])


if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, port=5039, host='127.0.0.1')
