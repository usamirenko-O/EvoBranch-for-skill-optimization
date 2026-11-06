"""OURS — the optimizer-agnostic outer search framework.

Nothing in this package imports an optimizer. SkillOpt-, SSO-, EvoSkill- and
SkillGen-specific behaviour lives behind ``ours.adapters``.
"""
from ours.beam import BeamManager, BeamSelection, TiePolicy, UnresolvedTiePolicy
from ours.controller import RoundResult, RoundTrace, SearchController
from ours.evaluation import PoolScorer, ScoringReport
from ours.operator import (
    NoOpOperator,
    OperatorProposal,
    PassThroughOperator,
    SkillOperator,
)
from ours.partition import (
    EvidencePartitioner,
    RoundRobinPartitioner,
    SingletonPartitioner,
)
from ours.pool import CandidatePool, PoolStage, RawProposalStore
from ours.search import SearchResult, run_search
from ours.types import (
    CandidateState,
    CostAccounting,
    Evidence,
    EvidenceSubset,
    Origin,
    RawProposal,
    Score,
    Stage,
)
from ours.workspace import CandidateIsolationError, RunWorkspace

__all__ = [
    "BeamManager", "BeamSelection", "TiePolicy", "UnresolvedTiePolicy",
    "CandidatePool", "PoolStage", "RawProposalStore",
    "CandidateState", "CostAccounting", "Evidence", "EvidenceSubset",
    "Origin", "RawProposal", "Score", "Stage",
    "EvidencePartitioner", "RoundRobinPartitioner", "SingletonPartitioner",
    "NoOpOperator", "OperatorProposal", "PassThroughOperator", "SkillOperator",
    "SearchResult", "run_search",
    "PoolScorer", "ScoringReport",
    "RoundResult", "RoundTrace", "SearchController",
    "CandidateIsolationError", "RunWorkspace",
]
