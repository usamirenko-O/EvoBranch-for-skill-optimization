"""Optimizer adapters. The only place optimizer-specific behaviour may live."""
from ours.adapters.base import EvaluatorAdapter, OptimizerAdapter, ProposalResult

__all__ = ["EvaluatorAdapter", "OptimizerAdapter", "ProposalResult"]
