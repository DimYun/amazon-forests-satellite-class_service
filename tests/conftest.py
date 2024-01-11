import os.path # noqa: WPS301

import cv2
import pytest
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.container_task import Container
import app as app_routs
from app import set_routers

TESTS_DIR = 'tests'


@pytest.fixture(scope='session')
def sample_image_bytes():
    f = open(os.path.join(TESTS_DIR, 'images', 'file_0.jpg'), 'rb')  # noqa: WPS515
    try:
        yield f.read()
    finally:
        f.close()


@pytest.fixture
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'images', 'file_0.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def app_config():
    return OmegaConf.load(os.path.join(TESTS_DIR, 'test_config.yml'))


@pytest.fixture
def app_container(app_config):
    container = Container()
    container.config.from_dict(app_config)
    return container


@pytest.fixture
def wired_app_container(app_config):
    container = Container()
    container.config.from_dict(app_config)
    container.wire([app_routs])
    yield container
    container.unwire()


@pytest.fixture
def test_app(wired_app_container):
    app = FastAPI()
    set_routers(app)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
