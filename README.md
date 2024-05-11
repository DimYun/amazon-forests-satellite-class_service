# Домашняя работа №1. Сервис


Сервис реализован на FastAPI и предназначен для выдачи предсказаний модели [многоклассовой классификации](https://gitlab.deepschool.ru/cvr-dec23/d.iunovidov/hw-01-modeling/-/tree/dev?ref_type=heads).


Адрес для тестов: http://91.206.15.25:5039

Документация и тестирование GET и POST запросов: http://91.206.15.25:5039/docs#/default/process_content_process_content_post


## Настройка окружения

Сначала создать и активировать venv:

```bash
python3 -m venv venv
. venv/bin/activate
```

Дальше поставить зависимости:

```bash
make install
```

### Команды

#### Подготовка
* `make install` - установка библиотек

#### Запуск сервиса
* `make run_app` - запустить сервис. Можно с аргументом `APP_PORT`

#### Сборка образа
* `make build` - собрать образ. Можно с аргументами `DOCKER_TAG`, `DOCKER_IMAGE`

#### Статический анализ
* `make lint` - запуск линтеров

#### Тестирование
* `make run_unit_tests` - запуск юнит-тестов
* `make run_integration_tests` - запуск интеграционных тестов
* `make run_all_tests` - запуск всех тестов

