import os
import typing as tp
import onnxruntime as ort
import numpy as np
import cv2
import torch
from PIL import Image

from dependency_injector import containers, providers


class Storage:
    def __init__(self, config: dict):
        self._config = config
        os.makedirs(config['dir_path'], exist_ok=True)

    def save(self, content_str: str, content_id: str):
        if not os.path.exists(self._get_path(content_id)):
            with open(self._get_path(content_id), 'w') as f:
                f.write(content_str)

    def get(self, content_id: str) -> tp.Optional[str]:
        content_path = self._get_path(content_id)
        if not os.path.exists(content_path):
            return 'Start process image first'
        with open(content_path, 'r') as f:
            return f.read()

    def _get_path(self, content_id: str):
        return os.path.join(self._config['dir_path'], content_id)


class ProcessPlanet:
    def __init__(self, storage: Storage):
        self._storage = storage
        self.names = [
            'haze',
            'primary',
            'agriculture',
            'clear',
            'water',
            'habitation',
            'road',
            'cultivation',
            'slash_burn',
            'cloudy',
            'partly_cloudy',
            'conventional_mine',
            'bare_ground',
            'artisinal_mine',
            'blooming',
            'selective_logging',
            'blow_down',
        ]
        os.makedirs('../uploaded_imgs', exist_ok=True)

    def process(self, image: bytes, content_id: str):
        scores_str = self._storage.get(content_id)
        if scores_str == 'Start process image first':
            ort_session = ort.InferenceSession(
                'src/onnx_planet_model.onnx',
                providers=['CPUExecutionProvider'],
            )

            # готовим входной тензор
            onnx_input = self.onnx_preprocessing(image)
            onnx_input = np.concatenate([onnx_input])

            ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
            # выполняем инференс ONNX Runtime
            ort_outputs = torch.tensor(ort_session.run(None, ort_inputs)[0])
            scores_onnx = torch.sigmoid(ort_outputs)[0].cpu().numpy()

            scores_str = ', '.join(
                [f'{n}: {s}' for s, n in zip(scores_onnx, self.names)],
            )

            self._storage.save(scores_str, content_id)
        return scores_str

    def onnx_preprocessing(
        self,
        image: np.ndarray,
        image_size: tp.Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Convert numpy-image to array for inference ONNX Runtime model.
        """

        # resize
        image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)

        # normalize
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
        denominator = np.reciprocal(std, dtype=np.float32)
        image = image.astype(np.float32)
        image -= mean
        image *= denominator

        # transpose
        image = image.transpose((2, 0, 1))[None]
        return image


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    store = providers.Factory(
        Storage,
        config=config.content_process,
    )

    content_process = providers.Singleton(
        ProcessPlanet,
        storage=store.provider(),
    )


# @inject
def task1():
    config = {
        'content_process': {
            'dir_path': 'test_dir',
        },
    }

    container = Container()
    container.config.from_dict(config)

    content_process = container.content_process()

    image = cv2.imread('test.jpg')[..., ::-1]
    Image.fromarray(image)

    content_process.process(
        image,
        'test.jpg',
    )


def task2():
    config = {
        'content_process': {
            'dir_path': 'test_dir',
        },
    }

    container = Container()
    container.config.from_dict(config)
    config_mock = {
        'content_process': {
            'dir_path': 'mock_test_dir',
        },
    }
    with container.config.override(config_mock):
        content_process = container.content_process()
        image = cv2.imread('test.jpg')[..., ::-1]
        Image.fromarray(image)

        content_process.process(
            image,
            'test_img',
        )


if __name__ == '__main__':
    task1()
    task2()
