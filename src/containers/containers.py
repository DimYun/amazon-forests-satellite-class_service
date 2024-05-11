from dependency_injector import containers, providers

from src.services.planet_classifier import ProcessPlanet
from src.services.preprocess_utils import MultilabelPredictor


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    predictor = providers.Singleton(
        MultilabelPredictor,
        config=config.model_parameters,
    )

    content_process = providers.Singleton(
        ProcessPlanet,
        predictor=predictor.provider(),
        config=config.model_parameters,
    )
