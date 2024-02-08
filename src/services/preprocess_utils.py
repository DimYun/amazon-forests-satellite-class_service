import typing as tp

import cv2
import numpy as np
import onnxruntime as ort
import torch


def onnx_preprocessing(
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


def use_onnx(onnx_input: np.array):
    ort_session = ort.InferenceSession(
        "models/onnx_planet_model.onnx",
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
    # выполняем инференс ONNX Runtime
    return torch.tensor(ort_session.run(None, ort_inputs)[0])
