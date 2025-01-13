# Type of Amazon forests satellite image. Service

FastAPI service for Amazon forests satellite image 
[NN multilabel classification](https://github.com/DimYun/amazon-forests-satellite-class_model). 
I made accent on "industrial quality" code with next technologies:

* FastAPI
* Gitlab CI/CD (test, deploy, destroy)
* DVC
* Docker
* Unit & Integration tests with coverage report
* Linters (flake8 + wemake)

**Disclaimers**:

* the project was originally crated and maintained in GitLab local instance, some repo functionality may be unavailable
* the project was created by me and me only as part of the CVRocket professional development course
* this project is my first "industry grade" service for NN, for more advanced code and features please see [car-plate projects](https://github.com/DimYun/car-plate_service)


Location for manual test:
* https://amazon_forest_api.lydata.duckdns.org
* docs https://amazon_forest_api.lydata.duckdns.org/docs#/default/process_content_process_content_post
* user: john_smith 
* password: 085636251932027


## Setup of environment

First, create and activate `venv`:
    ```bash
    python3 -m venv venv
    . venv/bin/activate
    ```

Next, install dependencies:
    ```bash
    make install
    ```

### Commands

#### Preparation
* `make install` - install python dependencies

#### Run service
* `make run_app` - run servie. You can define argument `APP_PORT`

#### Build docker
* `make build` - you can define arguments `DOCKER_TAG`, `DOCKER_IMAGE`

#### Static analyse
* `make lint` - run linters

#### Tests
* `make run_unit_tests` - run unit tests
* `make run_integration_tests` - run integration tests
* `make run_all_tests` - run all tests

