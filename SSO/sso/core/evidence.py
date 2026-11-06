"""Evidence Aggregation and Ranking (paper eq. 9-14)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

from sso.core.types import (
    Activation,
    BehaviorCluster,
    BehaviorUnit,
    ClusterScore,
    ExecutionRef,
    Intent,
    PairwiseDecision,
)

# Numeric encoding of the two *clear* activation states, used only once both
# activations being compared are known to be clear (never applied to
# UNCLEAR -- see changed_cluster_set).
_NUMERIC = {Activation.PRESENT: 1, Activation.ABSENT: 0}


def resolve_cluster_activation(cluster: BehaviorCluster, execution: ExecutionRef) -> Activation:
    """alpha_ic(r): the common clear activation for cluster c at execution r.

    PAPER_SPECIFIED (Methodology, "Evidence Aggregation and Ranking"):
    "alpha_ic(r) is the common clear activation assigned by the task-local
    behavior records in cluster c. ... alpha_ic(r) = bot if no such record
    exists, the activation is unclear, or the records disagree."

    A `BehaviorRecord` missing an entry for `execution` in its
    `activations` map is treated as an UNCLEAR vote from that record
    (defensive default; not itself a paper case -- see docs/assumptions.md).
    """
    records = cluster.records_for_task(execution.task_id)
    votes: Set[Activation] = {record.activations.get(execution, Activation.UNCLEAR) for record in records}
    if len(votes) != 1:
        # len == 0: no task-local record in this cluster for this task.
        # len > 1: records disagree.
        # Either way the paper says alpha_ic(r) = bot.
        return Activation.UNCLEAR
    (only_vote,) = votes
    return only_vote


def changed_cluster_set(
    clusters: Sequence[BehaviorCluster], task_id: str, probe_index: int
) -> List[BehaviorCluster]:
    """A_i^k (eq. 9): clusters whose behavior clearly differs between the
    probe execution r_i^k and the anchor execution r_i^o for task i.

    PAPER_SPECIFIED. Both alpha_ic(r_i^k) and alpha_ic(r_i^o) must be clear
    (PRESENT or ABSENT) and must differ; UNCLEAR on either side excludes the
    cluster, it is never coerced to ABSENT/0.
    """
    anchor = ExecutionRef.anchor(task_id)
    probe = ExecutionRef.probe(task_id, probe_index)
    changed = []
    for cluster in clusters:
        a_probe = resolve_cluster_activation(cluster, probe)
        a_anchor = resolve_cluster_activation(cluster, anchor)
        if a_probe in _NUMERIC and a_anchor in _NUMERIC and a_probe != a_anchor:
            changed.append(cluster)
    return changed


def comparison_evidence(
    decision: PairwiseDecision, clusters: Sequence[BehaviorCluster]
) -> Dict[str, float]:
    """e_ic^k for a single decision, keyed by cluster_id (eq. 10-12).

    PAPER_SPECIFIED. Returns {} (no evidence assigned to any cluster) when:
      - the decision is a tie (y_i^k = 0): "Tie judge decisions ... contribute
        no evidence" is required by the task brief and is consistent with
        eq. 11, which is only defined "for a decisive comparison".
      - the changed-cluster set A_i^k is empty: the paper calls this
        "left unattributed", never divided by zero and never fabricated as
        zero evidence spread over every cluster.

    When A_i^k is non-empty, this satisfies eq. 12 by construction:
    sum_{c in A_i^k} |e_ic^k| == 1.
    """
    if decision.y == 0:
        return {}
    changed = changed_cluster_set(clusters, decision.task_id, decision.probe_index)
    if not changed:
        return {}
    anchor = ExecutionRef.anchor(decision.task_id)
    probe = ExecutionRef.probe(decision.task_id, decision.probe_index)
    n = len(changed)
    evidence: Dict[str, float] = {}
    for cluster in changed:
        d = _NUMERIC[resolve_cluster_activation(cluster, probe)] - _NUMERIC[
            resolve_cluster_activation(cluster, anchor)
        ]
        evidence[cluster.cluster_id] = decision.y * d / n
    return evidence


def aggregate_evidence(
    decisions: Sequence[PairwiseDecision], clusters: Sequence[BehaviorCluster]
) -> Dict[str, ClusterScore]:
    """Q_c and E_c across all decisions (eq. 13), plus intent (eq. 14).
    
    PAPER_SPECIFIED for Q_c, E_c, and intent-from-sign(Q_c). Clusters that
    never receive any evidence (never appear in any A_i^k) are simply absent
    from the result, which is behaviorally identical to the paper's
    "discard Q_c = 0" rule for ranking purposes.
    
    "contributing_tasks" operationalizes the paper's
    ranking tie-breaker phrase "evidence from more distinct tasks" as: a
    task contributes to cluster c if at least one of its decisions assigned
    c nonzero evidence. The paper names this tie-breaker but does not spell
    out this definition; see docs/assumptions.md."""
    q: Dict[str, float] = defaultdict(float)
    e: Dict[str, float] = defaultdict(float)
    contributing_tasks: Dict[str, Set[str]] = defaultdict(set)
    by_id = {cluster.cluster_id: cluster for cluster in clusters}

    for decision in decisions:
        for cluster_id, value in comparison_evidence(decision, clusters).items():
            q[cluster_id] += value
            e[cluster_id] += abs(value)
            contributing_tasks[cluster_id].add(decision.task_id)

    scores: Dict[str, ClusterScore] = {}
    for cluster_id, q_c in q.items():
        if q_c > 0:
            intent = Intent.ENCOURAGE
        elif q_c < 0:
            intent = Intent.SUPPRESS
        else:
            intent = Intent.NEUTRAL
        scores[cluster_id] = ClusterScore(
            cluster_id=cluster_id,
            description=by_id[cluster_id].representative_description,
            q=q_c,
            e=e[cluster_id],
            contributing_tasks=frozenset(contributing_tasks[cluster_id]),
            intent=intent,
        )
    return scores


def rank_clusters(scores: Sequence[ClusterScore]) -> List[ClusterScore]:
    """Rank non-neutral clusters by |Q_c| desc, then E_c desc, then distinct-contributing-task-count desc."""
    non_neutral = [score for score in scores if score.intent != Intent.NEUTRAL]
    return sorted(
        non_neutral,
        key=lambda s: (-abs(s.q), -s.e, -s.contributing_task_count, s.cluster_id),
    )


def select_behavior_units(ranked: Sequence[ClusterScore], update_budget: int) -> List[BehaviorUnit]:
    """U_t: the top `update_budget` ranked clusters (eq. 14, "Retain up to U
    highest-ranked clusters as the behavior units U_t").

    PAPER_SPECIFIED. `update_budget` corresponds to the paper's U (U=4 in
    the reported experiments; the value itself is a run configuration, not
    hardcoded here).
    """
    if update_budget < 0:
        raise ValueError("update_budget must be >= 0")
    return [
        BehaviorUnit(
            cluster_id=score.cluster_id,
            description=score.description,
            intent=score.intent,
            q=score.q,
            e=score.e,
            contributing_task_count=score.contributing_task_count,
        )
        for score in ranked[:update_budget]
    ]


def compute_behavior_units(
    decisions: Sequence[PairwiseDecision],
    clusters: Sequence[BehaviorCluster],
    update_budget: int,
) -> List[BehaviorUnit]:
    """Aggregate decisions and clusters into ranked behavior units for skill rendering."""
    scores = aggregate_evidence(decisions, clusters)
    ranked = rank_clusters(list(scores.values()))
    return select_behavior_units(ranked, update_budget)
