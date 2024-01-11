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
#
#
# def test_predict_proba(client: TestClient, sample_image_bytes: bytes):
#     files = {
#         'image': sample_image_bytes,
#     }
#     response = client.post('/poster/predict_proba', files=files)
#
#     assert response.status_code == HTTPStatus.OK
#
#     genre2prob = response.json()
#
#     for genre_prob in genre2prob.values():
#         assert genre_prob <= 1
#         assert genre_prob >= 0
