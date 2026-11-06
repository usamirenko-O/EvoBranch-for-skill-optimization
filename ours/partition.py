"""Partition native evidence with seeded K-means and optional history-aware grouping."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ours import history as _history
from ours.registry import register
from ours.resources import RunBound
from ours.types import Evidence, EvidenceSubset


class EvidencePartitioner(RunBound, ABC):
    """Splits one parent's evidence into the subsets proposals condition on."""

    name: str = "abstract"
    is_placeholder: bool = False

    @abstractmethod
    def partition_task_ids(
        self,
        task_ids: Sequence[str],
        features: dict[str, dict[str, Any]],
        *,
        k: int,
    ) -> list[tuple[str, ...]]:
        """Group task ids into at most *k* non-empty groups.

        Implementations see only task ids and the generic feature view, never
        an optimizer payload.
        """

    def report(self) -> dict | None:
        """What the last :meth:`partition` call did, for the run trace (seed,
        embedding model, cluster labels, ...). None when there is nothing to
        report (the trivial placeholders)."""
        return getattr(self, "last_report", None)

    def partition(
        self,
        evidence: Evidence,
        features: dict[str, dict[str, Any]],
        *,
        k: int,
        materialise,
    ) -> list[EvidenceSubset]:
        """Group task ids, then ask the adapter to materialise each subset.

        ``materialise(subset_id, task_ids, partitioner) -> EvidenceSubset`` is
        normally ``OptimizerAdapter.slice_evidence``.
        """
        groups = self.partition_task_ids(list(evidence.task_ids), features, k=k)
        groups = [g for g in groups if g]
        return [
            materialise(
                subset_id=f"{evidence.evidence_id}:e{idx}",
                task_ids=group,
                partitioner=self.name,
            )
            for idx, group in enumerate(groups)
        ]


@register("partitioner", "round_robin_placeholder")
class RoundRobinPartitioner(EvidencePartitioner):
    """PLACEHOLDER. Deterministic round-robin over sorted task ids.

    Not a scientific clustering method and not intended to become one. It
    guarantees only: deterministic, non-empty groups, every task used exactly
    once -- enough to drive and test the control flow offline.
    """

    name = "round_robin_placeholder"
    is_placeholder = True

    def partition_task_ids(self, task_ids, features, *, k):
        ordered = sorted(str(t) for t in task_ids)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        buckets: list[list[str]] = [[] for _ in range(min(k, len(ordered)) or 1)]
        for i, tid in enumerate(ordered):
            buckets[i % len(buckets)].append(tid)
        return [tuple(b) for b in buckets]


@register("partitioner", "singleton_placeholder")
class SingletonPartitioner(EvidencePartitioner):
    """PLACEHOLDER. One subset containing all evidence (K=1).

    Useful as the degenerate control: with this partitioner and a no-op
    operator, one OURS round proposes exactly once from the full evidence,
    which is the same conditioning the native single-candidate loop uses.
    """

    name = "singleton_placeholder"
    is_placeholder = True

    def partition_task_ids(self, task_ids, features, *, k):
        ordered = tuple(sorted(str(t) for t in task_ids))
        return [ordered] if ordered else []


@register("partitioner", "kmeans_native_text")
class NativeTextKMeansPartitioner(EvidencePartitioner):
    """Cluster embeddings of native evidence text with seeded K-means.
    
    Return at most k nonempty groups, preserving evidence order within each group."""

    name = "kmeans_native_text"
    is_placeholder = False

    #: Fixed number of k-means++ restarts; the best inertia wins. Part of the
    #: recorded configuration, not a tuning knob.
    N_INIT = 10

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int:
            raise ValueError(
                "kmeans_native_text needs an explicit integer seed "
                "(design: partitioner_options={\"seed\": <int>})"
            )
        self.seed = seed
        self.last_report: dict | None = None

    def partition_task_ids(self, task_ids, features, *, k):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ordered = [str(t) for t in task_ids]
        texts = []
        for tid in ordered:
            view = features.get(tid) or {}
            if "native_text" not in view:
                raise ValueError(
                    f"task {tid!r} has no native_text; the adapter must serialise its "
                    "native evidence for the partitioner"
                )
            texts.append(str(view["native_text"]))
        n = len(ordered)
        k_eff = min(k, n)
        report = {
            "partitioner": self.name,
            "seed": self.seed,
            "n_init": self.N_INIT,
            "k_requested": int(k),
            "n_evidence": n,
            "k_effective": k_eff,
            "embedding_model": None,
            "embedded": False,
            "labels": {},
            "n_groups": 0,
            "dropped_empty_groups": 0,
        }
        if n == 0:
            self.last_report = report
            return []
        if k_eff == 1:
            report["labels"] = {tid: 0 for tid in ordered}
            report["n_groups"] = 1
            self.last_report = report
            return [tuple(ordered)]

        import numpy as np
        from sklearn.cluster import KMeans

        resources = self.require_resources("the k-means partitioner (embeddings)")
        from ours.resources import ROLE_PARTITIONER

        vectors = resources.embed(texts, role=ROLE_PARTITIONER)
        if len(vectors) != n:
            raise RuntimeError(f"embedding returned {len(vectors)} vectors for {n} texts")
        matrix = np.asarray(vectors, dtype=float)
        labels = KMeans(n_clusters=k_eff, random_state=self.seed, n_init=self.N_INIT).fit_predict(matrix)
        groups: dict[int, list[str]] = {}
        for tid, label in zip(ordered, labels):
            groups.setdefault(int(label), []).append(tid)   # native order kept inside
        # order groups by first appearance in the native evidence order
        ordered_groups = sorted(groups.values(), key=lambda g: ordered.index(g[0]))
        report.update({
            "embedding_model": resources.embedding_model,
            "embedded": True,
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
            "labels": {tid: int(label) for tid, label in zip(ordered, labels)},
            "n_groups": len(ordered_groups),
            "dropped_empty_groups": k_eff - len(ordered_groups),
        })
        self.last_report = report
        return [tuple(g) for g in ordered_groups]


@register("partitioner", "history_aware_kmeans")
class HistoryAwareKMeansPartitioner(NativeTextKMeansPartitioner):
    """Group recurring unresolved issues together, then cluster the remaining evidence.
    
    Without recurring issues, apply K-means to all evidence. Each task appears once."""

    name = "history_aware_kmeans"
    is_placeholder = False

    def partition_task_ids(self, task_ids, features, *, k):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ordered = [str(t) for t in task_ids]
        recurring = [t for t in ordered
                     if (features.get(t) or {}).get("issue_status") == _history.STATUS_RECURRING]

        # No recurring-and-unresolved tasks (round 0 / no history), or K == 1:
        # exactly the frozen K-means over all tasks into k groups.
        if not recurring or k == 1:
            groups = super().partition_task_ids(ordered, features, k=k)
            self.last_report = self._history_report(
                base=self.last_report, recurring_group=(), recurring_issue_ids=[],
                n_recurring=len(recurring), mode="plain_kmeans", n_groups=len(groups),
            )
            return groups

        # The recurring tasks are one group; the rest go through K-means into
        # k - 1 groups (native evidence order preserved inside the recurring
        # group, as everywhere else).
        recurring_set = set(recurring)
        recurring_group = tuple(t for t in ordered if t in recurring_set)
        remaining = [t for t in ordered if t not in recurring_set]
        sub_groups = super().partition_task_ids(remaining, features, k=k - 1) if remaining else []
        base = self.last_report if remaining else None
        issue_ids: list[str] = []
        for t in recurring_group:
            for iid in (features.get(t) or {}).get("issue_ids") or []:
                if iid not in issue_ids:
                    issue_ids.append(str(iid))
        groups = [recurring_group] + [tuple(g) for g in sub_groups]
        self.last_report = self._history_report(
            base=base, recurring_group=recurring_group, recurring_issue_ids=issue_ids,
            n_recurring=len(recurring_group), mode="recurring_group_plus_kmeans",
            n_groups=len(groups),
        )
        return groups

    def _history_report(self, *, base, recurring_group, recurring_issue_ids, n_recurring,
                        mode, n_groups) -> dict:
        report = dict(base or {})
        report.update({
            "partitioner": self.name,
            "seed": self.seed,
            "history_aware_mode": mode,
            "recurring_group_task_ids": list(recurring_group),
            "recurring_issue_ids": recurring_issue_ids,
            "n_recurring_tasks": n_recurring,
            "n_groups": n_groups,
        })
        return report
