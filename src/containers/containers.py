from dependency_injector import containers, providers
from src.services.planet_classifier import Storage, ProcessPlanet


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
