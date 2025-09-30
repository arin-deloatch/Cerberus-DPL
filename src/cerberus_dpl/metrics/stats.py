"""Cluster level statistics"""

import math
from typing import Dict, List, Union
from cerberus_dpl.models import DocRecord, TopicStats, Number


def _median(values: List[Number]) -> float:
    """
    Calculate the median of a list of numeric values.

    Parameters
    ----------
    values : List[Number]
        List of numeric values to calculate median for.

    Returns
    -------
    float
        The median value. Returns 0.0 for empty lists.
    """
    count = len(values)
    if count == 0:
        return 0.0
    sorted_values = sorted(values)
    middle_index = count // 2
    return (
        float(sorted_values[middle_index])
        if count % 2
        else (sorted_values[middle_index - 1] + sorted_values[middle_index]) / 2.0
    )


def _percentile(values: List[Number], percentile: float) -> float:
    """
    Calculate a specific percentile of a list of integers.

    Parameters
    ----------
    values : List[Number]
        List of integer values to calculate percentile for.
    percentile : float
        Percentile to calculate (e.g., 0.75 for 75th percentile).

    Returns
    -------
    float
        The calculated percentile value. Returns 0.0 for empty lists.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = percentile * (len(sorted_values) - 1)
    lower_index, upper_index = int(index), min(int(index) + 1, len(sorted_values) - 1)
    fraction = index - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _iqr(values: List[Number]) -> float:
    """
    Calculate the interquartile range (IQR) of a list of integers.

    The IQR is the difference between the 75th and 25th percentiles,
    representing the middle 50% of the data distribution.

    Parameters
    ----------
    values : List[Number]
        List of integer values to calculate IQR for.

    Returns
    -------
    float
        The interquartile range (Q3 - Q1).
    """
    return _percentile(values, 0.75) - _percentile(values, 0.25)


def _mad(values: List[Number]) -> float:
    """
    Calculate the median absolute deviation (MAD) of a list of integers.

    MAD is a robust measure of variability that represents the median
    of the absolute deviations from the data's median.

    Parameters
    ----------
    values : List[Number]
        List of integer values to calculate MAD for.

    Returns
    -------
    float
        The median absolute deviation.
    """
    median_value = _median(values)
    absolute_deviations = [abs(value - median_value) for value in values]
    return _median(absolute_deviations)


def summarize_by_topic(document_records: List[DocRecord]) -> List[TopicStats]:
    """
    Compute per-topic statistics of token lengths.

    Parameters
    ----------
    document_records : List[DocRecord]
        List of documents with assigned topics and token lengths.

    Returns
    -------
    List[TopicStats]
        One entry per topic with mean, median, min, max, std, IQR, and MAD.
    """
    topic_to_token_lengths: Dict[int, List[Union[int, float]]] = {}

    for document_record in document_records:
        topic_to_token_lengths.setdefault(document_record.topic, []).append(
            document_record.token_len
        )

    topic_statistics: List[TopicStats] = []
    for topic_id, token_lengths in sorted(
        topic_to_token_lengths.items(), key=lambda item: item[0]
    ):
        document_count = len(token_lengths)
        mean_length = sum(token_lengths) / document_count if document_count else 0.0
        variance = (
            (
                sum((length - mean_length) ** 2 for length in token_lengths)
                / (document_count - 1)
            )
            if document_count > 1
            else 0.0
        )
        topic_statistics.append(
            TopicStats(
                topic=topic_id,
                n_docs=document_count,
                mean_len=mean_length,
                median_len=_median(token_lengths),
                min_len=float(min(token_lengths)) if document_count else 0.0,
                max_len=float(max(token_lengths)) if document_count else 0.0,
                std_len=math.sqrt(variance),
                iqr=_iqr(token_lengths),
                mad=_mad(token_lengths),
            )
        )
    return topic_statistics


if __name__ == "__main__":
    print(_iqr([1, 32, 3, 5.2]))
