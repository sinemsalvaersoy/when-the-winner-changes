"""Tools for auditing the reliability of scientific ML benchmark claims."""

from .analysis import compare_sources, metric_winners, rerun_winner_probabilities

__all__ = ["compare_sources", "metric_winners", "rerun_winner_probabilities"]
__version__ = "0.1.0"

