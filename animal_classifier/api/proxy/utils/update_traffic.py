from typing import Dict

from animal_classifier.api.proxy.base_types.proxy_config import ProxyConfig
from animal_classifier.cfg import cfg


def is_it_time_to_assess(prediction_summary: Dict[int, Dict[str, int]], current_version: int) -> bool:
    """
    Check if it is time to assess the performance of the current model.
    Our scheduler evaluates every COUNT_TRAFFIC_UPDATE requests. It only
    considers increasing traffic, so after hitting 100% traffic to current
    versions, the scheduler will not be triggered.

    Args:
        prediction_summary: The prediction summary.
        current_version: The current version.

    Returns:
        True if it is time to assess the performance of the current model, False otherwise.
    """
    total_predictions = prediction_summary[current_version]["total"]
    idx_traffic_steps, remainder = divmod(total_predictions, cfg.COUNT_TRAFFIC_UPDATE)
    # if we hit the target count, we are ready to increase traffic if performance improved
    if (total_predictions > 0) and (remainder == 0) and (idx_traffic_steps < len(cfg.TRAFFIC_STEPS)):
        return True
    return False


def get_model_accuracy(prediction_summary: Dict[int, Dict[str, int]], version: int) -> float:
    """
    Get the accuracy of the model.

    Args:
        prediction_summary: The prediction summary.
        version: The version.

    Returns:
        The accuracy of the model.
    """
    return (
        prediction_summary[version]["correct"] / prediction_summary[version]["total"]
        if prediction_summary[version]["total"] > 0
        else 0
    )


def does_performance_hold_up(prediction_summary: Dict[int, Dict[str, int]], config: ProxyConfig) -> bool:
    """
    Check if the performance of the current model holds up.
    We compare the accuracy of the current model with the previous model.

    Args:
        prediction_summary: The prediction summary.
        config: The current configuration.

    Returns:
        True if the performance of the current model holds up, False otherwise.
    """
    accuracy_previous = get_model_accuracy(prediction_summary=prediction_summary, version=config.previous_version)
    accuracy_current = get_model_accuracy(prediction_summary=prediction_summary, version=config.current_version)
    if accuracy_current >= accuracy_previous:
        return True
    return False


def update_traffic(config: ProxyConfig, prediction_summary: Dict[int, Dict[str, int]]) -> ProxyConfig:
    """
    Explore updating the traffic based on the prediction summary.

    Args:
        config: The current configuration.
        prediction_summary: The prediction summary.

    Returns:
        The updated configuration.
    """

    if config.current_version != config.previous_version:
        if is_it_time_to_assess(prediction_summary=prediction_summary, current_version=config.current_version):
            if does_performance_hold_up(prediction_summary=prediction_summary, config=config):
                config.increase_traffic()
            else:
                config = ProxyConfig(current_version=config.previous_version, previous_version=config.previous_version)
    return config
