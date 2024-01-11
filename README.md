### DI & FastAPI

В `container_task.py` лежит код двух классов:

* `Storage` - умеет сохранять контент в папку 
* `Downloader` - умеет скачивать контент по url и звать `storage`, чтобы тот сохранил контент

В `fastapi_task.py` лежит приложение FastAPI, у которого две ручки:
1. `get_content` - возвращает контент по `content_id`
2. `save_content` - сохраняет контент, который находится по `content_url`

Пример работы можно посмотреть в `if __name__ == '__main__` в `container_task.py`

