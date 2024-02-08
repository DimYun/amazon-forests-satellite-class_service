import os
import typing as tp

import cv2
import numpy as np
import onnxruntime as ort
import torch


class Storage:
    def __init__(self, config: dict):
        self._config = config
        os.makedirs(config["dir_path"], exist_ok=True)
        os.makedirs(config["dir_upload"], exist_ok=True)

    def save(self, content_str: str, content_id: str):
        if not os.path.exists(self._get_path(content_id)):
            with open(self._get_path(content_id), "w") as save_file:
                save_file.write(content_str)

    def get(self, content_id: str) -> tp.Optional[str]:
        content_path = self._get_path(content_id)
        if not os.path.exists(content_path):
            return "Start process image first"
        with open(content_path, "r") as load_file:
            return load_file.read()

    def _get_path(self, content_id: str):
        return os.path.join(self._config["dir_path"], content_id)


class ProcessPlanet:
    STATUS_READY = "Start process image first"
    def __init__(self, storage: Storage):
        self._storage = storage
        self.names = [
            "haze",
            "primary",
            "agriculture",
            "clear",
            "water",
            "habitation",
            "road",
            "cultivation",
            "slash_burn",
            "cloudy",
            "partly_cloudy",
            "conventional_mine",
            "bare_ground",
            "artisinal_mine",
            "blooming",
            "selective_logging",
            "blow_down",
        ]

    def process(self, image: bytes, content_id: str):
        scores_str = self._storage.get(content_id)
        if scores_str == self.STATUS_READY:
            # готовим входной тензор
            onnx_input = self.onnx_preprocessing(image)
            onnx_input = np.concatenate([onnx_input])
            scores_onnx = torch.sigmoid(self.use_onnx(onnx_input))[0]
            scores_onnx = scores_onnx.cpu().numpy()
            scores_str = [
                f"{names}: {model_scores}" for model_scores, names in zip(
                    scores_onnx, self.names
                )
            ]
            scores_str = ", ".join(scores_str)
            self._storage.save(scores_str, content_id)
        return scores_str

    def use_onnx(self, onnx_input: np.array):
        ort_session = ort.InferenceSession(
            "models/onnx_planet_model.onnx",
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
        # выполняем инференс ONNX Runtime
        return torch.tensor(ort_session.run(None, ort_inputs)[0])

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
        return image.transpose((2, 0, 1))[None]