"""Deterministic, paper-specified core of SSO (Self-Supervised Skill Optimization).

This package holds only the parts of the SSO algorithm (Peng et al., "Self-
Supervised Skill Optimization", arXiv:2607.28777) that are pure data and pure
math: the data structures behavior extraction/clustering would produce
(`types.py`) and the Evidence Aggregation and Ranking step, eq. 9-14
(`evidence.py`). It intentionally contains no LLM calls, no benchmark
adapters, and no sampling policy -- those are later phases.
"""
