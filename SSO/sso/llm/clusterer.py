"""Cluster extracted behavior records for evidence aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

from sso.core.types import BehaviorCluster, BehaviorRecord
from sso.llm.client import LLMClient
from sso.llm._structured import (
    LLMResponseError, array, ask_json_validated, object_keys, text,
)


@dataclass(frozen=True)
class ClusterAssignment:
    """One cluster in C_t (eq. 8), expressed purely in terms of record ids."""

    representative_record_id: str
    member_record_ids: FrozenSet[str]

    def __post_init__(self) -> None:
        if self.representative_record_id not in self.member_record_ids:
            raise ValueError(
                "representative_record_id must be one of member_record_ids "
                f"(got {self.representative_record_id!r} not in "
                f"{sorted(self.member_record_ids)!r})"
            )


class BehaviorClusterer(ABC):
    """f_cluster: groups semantically equivalent behavior descriptions."""

    @abstractmethod
    def cluster(self, descriptions: Sequence[Tuple[str, str]]) -> List[ClusterAssignment]:
        """Partition behavior descriptions into clusters."""
        raise NotImplementedError


def assemble_clusters(
    assignments: Sequence[ClusterAssignment], records: Sequence[BehaviorRecord]
) -> List[BehaviorCluster]:
    """Reattach full `BehaviorRecord` data (activations included) to the id-only output of `BehaviorClusterer.cluster`, producing the `BehaviorCluster` objects `sso/core/evidence.py` operates on."""
    by_id: Dict[str, BehaviorRecord] = {record.record_id: record for record in records}
    clusters: List[BehaviorCluster] = []
    for assignment in assignments:
        members = [by_id[record_id] for record_id in assignment.member_record_ids]
        representative = by_id[assignment.representative_record_id]
        clusters.append(
            BehaviorCluster(
                cluster_id=assignment.representative_record_id,
                representative_description=representative.description,
                member_records=members,
            )
        )
    return clusters


# PAPER_SPECIFIED: eq. 8 and Algorithm 1 line 15: description-only semantic
# partition, related-but-distinct behaviors separate, original representative.
# REPRODUCTION_CHOICE: exact prompt and id-only JSON assignments.
_CLUSTER_PROMPT = """Partition the supplied behavior descriptions into
clusters. Group descriptions only when they express the same observable
behavior. Keep related but distinct behaviors separate. Include every input
record id exactly once across all clusters. Choose one member record id as
each cluster's representative; do not write a summary or new description.
Return {"clusters": [{"representative_record_id": "...",
"member_record_ids": ["..."]}]}.
Each cluster must be nonempty. With no input descriptions return
{"clusters": []}.
"""


class LLMBehaviorClusterer(BehaviorClusterer):
    """Concrete f_cluster; only ids and descriptions enter the LLM request.
    
    enforce a full disjoint partition and member representative.
    reject duplicate input ids and malformed partitions
    rather than repairing them. Preserve response cluster order. Call the
    unchanged assemble_clusters helper separately with the original records."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def cluster(self, descriptions: Sequence[Tuple[str, str]]) -> List[ClusterAssignment]:
        data = []
        ids = set()
        for record_id, description in descriptions:
            if (not isinstance(record_id, str) or not record_id.strip()
                    or not isinstance(description, str) or not description.strip()):
                raise ValueError("record ids and descriptions must be nonempty text")
            if record_id in ids:
                raise ValueError("duplicate input record id")
            ids.add(record_id)
            data.append({"record_id": record_id, "description": description})

        def build(response):
            obj = object_keys(response, ("clusters",))
            seen = set()
            assignments = []
            for item in array(obj["clusters"]):
                cluster = object_keys(item, ("representative_record_id", "member_record_ids"))
                representative = text(cluster["representative_record_id"])
                members = [text(value) for value in array(cluster["member_record_ids"])]
                member_ids = frozenset(members)
                if (not members or len(member_ids) != len(members)
                        or not member_ids <= ids or seen & member_ids
                        or representative not in member_ids):
                    raise LLMResponseError("invalid cluster partition or representative")
                seen.update(member_ids)
                assignments.append(ClusterAssignment(representative, member_ids))
            if seen != ids:
                raise LLMResponseError("cluster partition omits input records")
            return assignments

        return ask_json_validated(self.client, _CLUSTER_PROMPT, {
            "descriptions": data,
        }, build)
