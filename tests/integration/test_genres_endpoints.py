from fastapi.testclient import TestClient
from http import HTTPStatus


def test_types_list(client: TestClient):
    response = client.get('/planet/get_content?content_id=test')
    assert response.status_code == HTTPStatus.OK

    planet_types = response.json()['content']

    assert isinstance(planet_types, str)


def test_predict(client: TestClient, sample_image_bytes: bytes):
    files = {
        'content_image': sample_image_bytes,
    }
    response = client.post('/planet/process_content', files=files)
    assert response.status_code == HTTPStatus.OK

    predicted_scores = response.json()['scores']

    assert isinstance(predicted_scores, str)
