import typing as tp

import cv2
import numpy as np
import onnxruntime as ort


class MultilabelPredictor:
    def __init__(self, config: dict):
        # Инициализировали один раз и сохранили в атрибут
        self.ort_session = ort.InferenceSession(
            config.checkpoint,
            providers=('CPUExecutionProvider',),
        )
        self.names = config.names

    def onnx_preprocessing(
        self,
        image: np.ndarray,
        image_size: tp.Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
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

    def predict(
        self,
        image: np.ndarray,
        image_size: tp.Tuple[int, int] = (224, 224),
    ) -> np.array:
        # готовим входной тензор
        onnx_input = self.onnx_preprocessing(
            image,
            image_size=image_size,
        )
        onnx_input = np.concatenate([onnx_input])
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: onnx_input,
        }
        # выполняем инференс ONNX Runtime
        ort_outputs = self.ort_session.run(None, ort_inputs)
        logits = ort_outputs[0]
        return self.sigmoid(logits).squeeze()

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-np.logaddexp(0, -x))

    def postprocess_predict_proba(self, predict: np.ndarray) -> tp.Dict[str, float]:
        """Постобработка для получения словаря с вероятностями.

        :param predict: вероятности после прогона модели;
        :return: словарь вида `жанр фильма`: вероятность.
        """
        return {
            self.names[int(i)]: float(predict[int(i)]) for i in predict.argsort()[::-1]
        }
