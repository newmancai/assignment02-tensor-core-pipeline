"""Diversity-preserving search primitives for a follow-up Stage 9 study.

This module is intentionally independent of the frozen v5 pilot runner.  It
turns role specialization into machine-checkable policy-space constraints and
provides deterministic local mutations around measured parents.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .stage9_policy import (
    ALLOWED_CPC_VALUES,
    PolicyExpression,
    PolicyFeature,
    PolicyLeaf,
    PolicyNode,
    PolicyPredicate,
    RuntimePolicyCandidate,
    canonical_policy_id,
    verify_policy,
)


class SearchLane(str, Enum):
    """Search niches that are expressible by the current runtime-policy IR."""

    WAVE_CAPACITY = "wave_capacity"
    TAIL_SKEW = "tail_skew"
    MIXED_INTERACTION = "mixed_interaction"
    CRITIC_SYNTHESIZER = "critic_synthesizer"


@dataclass(frozen=True)
class Mutation:
    operator: str
    path: tuple[str, ...]
    parent_policy_id: str
    candidate: RuntimePolicyCandidate


@dataclass(frozen=True)
class MeasuredPolicy:
    candidate: RuntimePolicyCandidate
    uniform_log_speedup: float
    correctness_passed: bool


def _walk(expression: PolicyExpression) -> tuple[PolicyExpression, ...]:
    if isinstance(expression, PolicyLeaf):
        return (expression,)
    return (expression, *_walk(expression.then_branch), *_walk(expression.else_branch))


def policy_features(candidate: RuntimePolicyCandidate) -> frozenset[PolicyFeature]:
    return frozenset(
        item.predicate.feature
        for item in _walk(candidate.root)
        if isinstance(item, PolicyNode)
    )


def lane_violations(
    lane: SearchLane, candidate: RuntimePolicyCandidate
) -> tuple[str, ...]:
    """Reject nominal roles that do not occupy their assigned search niche."""

    features = policy_features(candidate)
    if lane is SearchLane.WAVE_CAPACITY:
        return () if features == {PolicyFeature.TOTAL_CHUNKS} else (
            "wave_capacity must use total_chunks predicates exclusively",
        )
    if lane is SearchLane.TAIL_SKEW:
        allowed = {PolicyFeature.MAX_CHUNKS, PolicyFeature.NUM_SEQUENCES}
        return () if features and features <= allowed else (
            "tail_skew must use max_chunks and/or num_sequences, never total_chunks",
        )
    if lane is SearchLane.MIXED_INTERACTION:
        tail = {PolicyFeature.MAX_CHUNKS, PolicyFeature.NUM_SEQUENCES}
        return () if PolicyFeature.TOTAL_CHUNKS in features and features & tail else (
            "mixed_interaction must combine total_chunks with a tail/skew feature",
        )
    if lane is SearchLane.CRITIC_SYNTHESIZER:
        return () if len(set(candidate.parent_ids)) >= 2 else (
            "critic_synthesizer must compose at least two distinct parents",
        )
    raise ValueError(f"unknown search lane: {lane}")


def policy_niche(candidate: RuntimePolicyCandidate) -> tuple[object, ...]:
    """Feature-map cell used by a small MAP-Elites-style archive."""

    nodes = [item for item in _walk(candidate.root) if isinstance(item, PolicyNode)]
    leaves = [item.chunks_per_cta for item in _walk(candidate.root) if isinstance(item, PolicyLeaf)]

    def cpc_band(value: int) -> str:
        if value <= 6:
            return "low"
        if value <= 12:
            return "mid"
        return "high"

    feature_signature = tuple(sorted({item.predicate.feature.value for item in nodes}))
    return (len(nodes), feature_signature, tuple(sorted({cpc_band(value) for value in leaves})))


def elite_archive(rows: Iterable[MeasuredPolicy]) -> dict[tuple[object, ...], MeasuredPolicy]:
    """Keep the best correct candidate in every semantic niche."""

    elites: dict[tuple[object, ...], MeasuredPolicy] = {}
    for row in rows:
        if not row.correctness_passed or verify_policy(row.candidate):
            continue
        niche = policy_niche(row.candidate)
        incumbent = elites.get(niche)
        if incumbent is None or (
            row.uniform_log_speedup,
            canonical_policy_id(row.candidate),
        ) > (
            incumbent.uniform_log_speedup,
            canonical_policy_id(incumbent.candidate),
        ):
            elites[niche] = row
    return elites


def _replace_at(
    expression: PolicyExpression,
    path: tuple[str, ...],
    replacement: PolicyExpression,
) -> PolicyExpression:
    if not path:
        return replacement
    if not isinstance(expression, PolicyNode):
        raise ValueError("mutation path enters a leaf")
    head, *tail = path
    if head == "then":
        return replace(
            expression,
            then_branch=_replace_at(expression.then_branch, tuple(tail), replacement),
        )
    if head == "else":
        return replace(
            expression,
            else_branch=_replace_at(expression.else_branch, tuple(tail), replacement),
        )
    raise ValueError(f"invalid mutation path component: {head}")


def _paths(expression: PolicyExpression, prefix: tuple[str, ...] = ()):
    yield prefix, expression
    if isinstance(expression, PolicyNode):
        yield from _paths(expression.then_branch, (*prefix, "then"))
        yield from _paths(expression.else_branch, (*prefix, "else"))


def threshold_catalog(
    profile_rows: Sequence[Mapping[str, int]],
) -> dict[PolicyFeature, tuple[int, ...]]:
    """Derive data-supported cut points without asking an LLM to guess integers."""

    output = {}
    for feature in PolicyFeature:
        values = sorted({int(row[feature.value]) for row in profile_rows})
        cuts = set(values)
        cuts.update((left + right) // 2 for left, right in zip(values, values[1:]))
        output[feature] = tuple(sorted(value for value in cuts if value > 0))
    return output


def deterministic_mutations(
    parent: RuntimePolicyCandidate,
    profile_rows: Sequence[Mapping[str, int]],
    *,
    lane: SearchLane | None = None,
) -> tuple[Mutation, ...]:
    """Enumerate one-edit, typed mutations and deduplicate them semantically."""

    parent_id = canonical_policy_id(parent)
    thresholds = threshold_catalog(profile_rows)
    allowed_cpc = sorted(ALLOWED_CPC_VALUES)
    candidates: list[Mutation] = []
    node_paths = [(path, item) for path, item in _paths(parent.root) if isinstance(item, PolicyNode)]

    # A collapsed incumbent cannot reach a different role niche through a
    # single threshold or leaf edit.  Add bounded, deterministic projections
    # that preserve tree shape while changing the physical feature family.
    assignments: list[tuple[PolicyFeature, ...]] = []
    if lane is SearchLane.TAIL_SKEW and node_paths:
        assignments.extend(
            tuple(feature for _ in node_paths)
            for feature in (PolicyFeature.MAX_CHUNKS, PolicyFeature.NUM_SEQUENCES)
        )
        if len(node_paths) > 1:
            assignments.append(tuple(
                PolicyFeature.MAX_CHUNKS if index % 2 == 0 else PolicyFeature.NUM_SEQUENCES
                for index in range(len(node_paths))
            ))
    elif lane is SearchLane.MIXED_INTERACTION and len(node_paths) > 1:
        for index in range(len(node_paths)):
            for feature in (PolicyFeature.MAX_CHUNKS, PolicyFeature.NUM_SEQUENCES):
                assignments.append(tuple(
                    feature if current == index else PolicyFeature.TOTAL_CHUNKS
                    for current in range(len(node_paths))
                ))

    for assignment in assignments:
        root = parent.root
        for (path, node), feature in zip(node_paths, assignment):
            source_values = thresholds[node.predicate.feature]
            target_values = thresholds[feature]
            source_index = min(
                range(len(source_values)),
                key=lambda index: (abs(source_values[index] - node.predicate.threshold), index),
            )
            target_index = round(
                source_index * (len(target_values) - 1) / max(1, len(source_values) - 1)
            )
            predicate = replace(
                node.predicate,
                feature=feature,
                threshold=target_values[target_index],
            )
            current = next(item for current_path, item in _paths(root) if current_path == path)
            root = _replace_at(root, path, replace(current, predicate=predicate))
        candidate = RuntimePolicyCandidate(root, parent_ids=(parent_id,))
        candidates.append(Mutation("lane_projection", (), parent_id, candidate))

    for path, item in _paths(parent.root):
        if isinstance(item, PolicyLeaf):
            index = allowed_cpc.index(item.chunks_per_cta)
            for neighbor in allowed_cpc[max(0, index - 1): index] + allowed_cpc[index + 1: index + 2]:
                root = _replace_at(parent.root, path, PolicyLeaf(neighbor))
                candidate = RuntimePolicyCandidate(root, parent_ids=(parent_id,))
                candidates.append(Mutation("adjacent_cpc", path, parent_id, candidate))
            continue
        catalog = thresholds[item.predicate.feature]
        if catalog:
            nearest = sorted(catalog, key=lambda value: (abs(value - item.predicate.threshold), value))[:3]
            for value in nearest:
                if value == item.predicate.threshold:
                    continue
                predicate = replace(item.predicate, threshold=value)
                root = _replace_at(parent.root, path, replace(item, predicate=predicate))
                candidate = RuntimePolicyCandidate(root, parent_ids=(parent_id,))
                candidates.append(Mutation("data_supported_threshold", path, parent_id, candidate))

    unique = {}
    for mutation in candidates:
        if verify_policy(mutation.candidate):
            continue
        if lane is not None and lane_violations(lane, mutation.candidate):
            continue
        policy_id = canonical_policy_id(mutation.candidate)
        if policy_id != parent_id:
            unique.setdefault(policy_id, mutation)
    return tuple(unique[key] for key in sorted(unique))
