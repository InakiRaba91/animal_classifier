from calendar import c
from functools import wraps
import re
from unittest.mock import patch
import pytest

from animal_classifier.api.proxy.utils import update_traffic
from animal_classifier.api.proxy.utils.update_traffic import does_performance_hold_up, get_model_accuracy, is_it_time_to_assess
from animal_classifier.cfg import cfg
from animal_classifier.api.proxy.service import ProxyConfig

NUM_TRAFFIC_STEPS = len(cfg.TRAFFIC_STEPS)


@pytest.mark.parametrize("total, expected_time_to_assess", [
    (0, False),
    (int(0.5 * cfg.COUNT_TRAFFIC_UPDATE), False),
    (cfg.COUNT_TRAFFIC_UPDATE, True),
    (int((NUM_TRAFFIC_STEPS - 1.5) * cfg.COUNT_TRAFFIC_UPDATE), False),
    ((NUM_TRAFFIC_STEPS - 1) * cfg.COUNT_TRAFFIC_UPDATE, True),
    (NUM_TRAFFIC_STEPS * cfg.COUNT_TRAFFIC_UPDATE, False),
])
def test_is_it_time_to_assess(total, expected_time_to_assess):
    # Test when it is time to assess
    current_version = 1
    prediction_summary = {current_version: {"correct": 0, "total": total}}

    # execute
    time_to_assess = is_it_time_to_assess(prediction_summary=prediction_summary, current_version=current_version)
    
    # assert
    assert time_to_assess == expected_time_to_assess


@pytest.mark.parametrize("correct, total, expected_accuracy", [
    (0, 0, 0),
    (0, 5, 0),
    (3, 5, 0.6),
    (5, 5, 1),
])
def test_get_model_accuracy(correct, total, expected_accuracy):
    # Test when the total is 0
    version = 1
    prediction_summary = {version: {"correct": correct, "total": total}}
    
    # execute
    accuracy = get_model_accuracy(prediction_summary=prediction_summary, version=version)
    
    # assert
    assert accuracy == expected_accuracy


@pytest.mark.parametrize("correct_current, expected_performance_holds_up", [
    (0, False),
    (3, True),
    (5, True),
])
def test_does_performance_hold_up(correct_current, expected_performance_holds_up):
    previous_version, current_version = 0, 1
    config = ProxyConfig(current_version=current_version, previous_version=previous_version)
    prediction_summary = {
        previous_version: {"correct": 3, "total": 5}, 
        current_version: {"correct": correct_current, "total": 5},
    }
    
    # execute
    performance_holds_up = does_performance_hold_up(prediction_summary=prediction_summary, config=config)
    
    # assert
    assert performance_holds_up == expected_performance_holds_up


@pytest.mark.parametrize("current_version, time_to_assess, performance_holds_up, increased_traffic", [
    (1, True, True, False),  # only one version reachable
    (2, False, True, False),  # not time to assess
    (2, True, False, False),  # performance does not hold up
    (2, True, True, True),  # performance holds up

])
def test_update_traffic(current_version, time_to_assess, performance_holds_up, increased_traffic):
    # setup
    previous_version = 1
    config = ProxyConfig(current_version=current_version, previous_version=previous_version)
    prediction_summary = {
        previous_version: {"correct": 3, "total": 5}, 
        current_version: {"correct": 3, "total": 5},
    }

    # execute
    with patch("animal_classifier.api.proxy.utils.update_traffic.is_it_time_to_assess", return_value=time_to_assess):
        with patch("animal_classifier.api.proxy.utils.update_traffic.does_performance_hold_up", return_value=performance_holds_up):
            with patch.object(ProxyConfig, "increase_traffic", wraps=config.increase_traffic) as mocked_increase_traffic:
                update_traffic(config=config, prediction_summary=prediction_summary)

    # assert
    expected_calls = 1 if increased_traffic else 0
    assert mocked_increase_traffic.call_count == expected_calls
    assert config.current_version == current_version
    assert config.previous_version == previous_version


def test_update_traffic_rolls_back_when_performance_does_not_hold_up():
    # setup
    time_to_assess = True
    performance_holds_up = False
    previous_version, current_version = 1, 2
    config = ProxyConfig(current_version=current_version, previous_version=previous_version)
    prediction_summary = {
        previous_version: {"correct": 3, "total": 5}, 
        current_version: {"correct": 3, "total": 5},
    }

    # execute
    with patch("animal_classifier.api.proxy.utils.update_traffic.is_it_time_to_assess", return_value=time_to_assess):
        with patch("animal_classifier.api.proxy.utils.update_traffic.does_performance_hold_up", return_value=performance_holds_up):
            with patch.object(ProxyConfig, "increase_traffic", wraps=config.increase_traffic) as mocked_increase_traffic:
                updated_config = update_traffic(config=config, prediction_summary=prediction_summary)

    # assert
    assert mocked_increase_traffic.call_count == 0
    assert updated_config.current_version == previous_version
    assert updated_config.previous_version == previous_version
    assert updated_config.traffic == 0