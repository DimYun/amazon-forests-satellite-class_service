from copy import deepcopy

import numpy as np

from src.containers.containers import Container


def test_predicts_not_fail(app_container: Container, sample_image_np: np.ndarray):
    planet_analytics = app_container.content_process()
    planet_analytics.process(sample_image_np)


def test_prob_less_or_equal_to_one(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    planet_analytics = app_container.content_process()
    planet2prob = planet_analytics.process(sample_image_np)
    for _, value_prob in planet2prob.items():
        prob = float(value_prob)
        assert prob <= 1
        assert prob >= 0


def test_predict_dont_mutate_initial_image(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    initial_image = deepcopy(sample_image_np)
    planet_analytics = app_container.content_process()
    planet_analytics.process(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)
