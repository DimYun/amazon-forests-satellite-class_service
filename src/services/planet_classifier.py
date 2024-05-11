from src.services.preprocess_utils import MultilabelPredictor


class ProcessPlanet:
    status_ready = "Start process image first"

    def __init__(
        self,
        predictor: MultilabelPredictor,
        config: dict,
    ):
        self.predictor = predictor
        self._config = config

    def process(self, image: bytes) -> dict:
        scores_onnx = self.predictor.predict(
            image,
            image_size=(self._config["img_width"], self._config["img_height"]),
        )
        return self.predictor.postprocess_predict_proba(scores_onnx)
