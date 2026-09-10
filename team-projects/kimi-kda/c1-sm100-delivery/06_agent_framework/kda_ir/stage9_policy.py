"""Typed runtime-policy search space for the Stage 9 ablation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import FrozenSet, Mapping, Tuple, Union

from .runtime_profile import RuntimeWorkloadProfile
from .verify import Diagnostic


class PolicyFeature(str, Enum):
    TOTAL_CHUNKS = "total_chunks"
    MAX_CHUNKS = "max_chunks"
    NUM_SEQUENCES = "num_sequences"
    RESIDENT_GRID_CAPACITY_CTAS = "resident_grid_capacity_ctas"


class PredicateOperator(str, Enum):
    LT = "lt"
    LE = "le"
    EQ = "eq"
    GE = "ge"
    GT = "gt"


ALLOWED_CPC_VALUES = frozenset(
    {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 20, 24, 32}
)
ALLOWED_POLICY_FEATURES = frozenset(PolicyFeature)


@dataclass(frozen=True)
class PolicyPredicate:
    feature: PolicyFeature
    operator: PredicateOperator
    threshold: int


@dataclass(frozen=True)
class PolicyLeaf:
    chunks_per_cta: int


@dataclass(frozen=True)
class PolicyNode:
    predicate: PolicyPredicate
    then_branch: "PolicyExpression"
    else_branch: "PolicyExpression"


PolicyExpression = Union[PolicyLeaf, PolicyNode]


@dataclass(frozen=True)
class PolicyEvaluationContext:
    total_chunks: int
    max_chunks: int
    num_sequences: int
    resident_grid_capacity_ctas: int

    def __post_init__(self) -> None:
        values = (
            self.total_chunks,
            self.max_chunks,
            self.num_sequences,
            self.resident_grid_capacity_ctas,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("policy evaluation features must be positive integers")
        if self.max_chunks > self.total_chunks:
            raise ValueError("max_chunks cannot exceed total_chunks")

    @classmethod
    def from_profile(
        cls,
        profile: RuntimeWorkloadProfile,
        *,
        resident_ctas_per_sm: int,
    ) -> "PolicyEvaluationContext":
        if resident_ctas_per_sm <= 0:
            raise ValueError("resident CTAs per SM must be positive")
        return cls(
            total_chunks=profile.total_chunks,
            max_chunks=profile.max_chunks,
            num_sequences=len(profile.seq_lens),
            resident_grid_capacity_ctas=profile.sm_count * resident_ctas_per_sm,
        )


@dataclass(frozen=True)
class LineageFragment:
    """A structural fragment claimed to be inherited from one parent."""

    parent_id: str
    subtree_id: str


@dataclass(frozen=True)
class RuntimePolicyCandidate:
    root: PolicyExpression
    parent_ids: Tuple[str, ...] = ()
    origin_proposal_ids: Tuple[str, ...] = ()
    lineage_fragments: Tuple[LineageFragment, ...] = ()


def _expression_payload(expression: PolicyExpression) -> dict:
    if isinstance(expression, PolicyLeaf):
        return {"kind": "leaf", "chunks_per_cta": expression.chunks_per_cta}
    if isinstance(expression, PolicyNode):
        feature = expression.predicate.feature
        operator = expression.predicate.operator
        return {
            "kind": "node",
            "predicate": {
                "feature": (
                    feature.value if isinstance(feature, PolicyFeature) else str(feature)
                ),
                "operator": (
                    operator.value
                    if isinstance(operator, PredicateOperator)
                    else str(operator)
                ),
                "threshold": expression.predicate.threshold,
            },
            "then": _expression_payload(expression.then_branch),
            "else": _expression_payload(expression.else_branch),
        }
    raise TypeError("policy expression must be a PolicyLeaf or PolicyNode")


def _payload_digest(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # Stage 9 artifacts are long-lived, externally audited receipts.  Keep the
    # complete digest: truncation is unnecessary here and weakens the identity
    # contract between the proposer, GPU evaluator, and certificate.
    return prefix + sha256(encoded.encode("utf-8")).hexdigest()


def policy_subtree_id(expression: PolicyExpression) -> str:
    return _payload_digest("kir-policy-subtree:", _expression_payload(expression))


def canonical_policy_id(candidate: RuntimePolicyCandidate) -> str:
    """Content-address policy semantics, excluding proposer and lineage wording."""

    return _payload_digest("kir-policy:", _expression_payload(candidate.root))


def _policy_depth(expression: PolicyExpression) -> int:
    if isinstance(expression, PolicyLeaf):
        return 0
    if isinstance(expression, PolicyNode):
        return 1 + max(
            _policy_depth(expression.then_branch),
            _policy_depth(expression.else_branch),
        )
    raise TypeError("policy expression must be a PolicyLeaf or PolicyNode")


def _walk(expression: PolicyExpression) -> Tuple[PolicyExpression, ...]:
    if isinstance(expression, PolicyLeaf):
        return (expression,)
    if isinstance(expression, PolicyNode):
        return (
            expression,
            *_walk(expression.then_branch),
            *_walk(expression.else_branch),
        )
    raise TypeError("policy expression must be a PolicyLeaf or PolicyNode")


def verify_policy(
    candidate: RuntimePolicyCandidate,
    *,
    max_depth: int = 2,
    allowed_features: FrozenSet[PolicyFeature] = ALLOWED_POLICY_FEATURES,
    allowed_cpc_values: FrozenSet[int] = ALLOWED_CPC_VALUES,
) -> Tuple[Diagnostic, ...]:
    """Validate the frozen Stage 9 policy language without timing evidence."""

    diagnostics = []
    if max_depth < 0:
        raise ValueError("maximum policy depth must be non-negative")
    try:
        depth = _policy_depth(candidate.root)
        expressions = _walk(candidate.root)
    except TypeError as exc:
        return (Diagnostic("KIR901", str(exc)),)
    if depth > max_depth:
        diagnostics.append(
            Diagnostic(
                "KIR902",
                "policy predicate depth %d exceeds maximum %d" % (depth, max_depth),
            )
        )
    for expression in expressions:
        if isinstance(expression, PolicyLeaf):
            value = expression.chunks_per_cta
            if isinstance(value, bool) or value not in allowed_cpc_values:
                diagnostics.append(
                    Diagnostic("KIR903", "unsupported chunks_per_cta leaf %r" % value)
                )
            continue
        predicate = expression.predicate
        if (
            not isinstance(predicate.feature, PolicyFeature)
            or predicate.feature not in allowed_features
        ):
            diagnostics.append(Diagnostic("KIR904", "unsupported policy feature"))
        if not isinstance(predicate.operator, PredicateOperator):
            diagnostics.append(Diagnostic("KIR905", "unsupported predicate operator"))
        if (
            isinstance(predicate.threshold, bool)
            or not isinstance(predicate.threshold, int)
            or predicate.threshold < 0
        ):
            diagnostics.append(
                Diagnostic("KIR906", "predicate threshold must be a non-negative integer")
            )
        if _expression_payload(expression.then_branch) == _expression_payload(
            expression.else_branch
        ):
            diagnostics.append(
                Diagnostic("KIR907", "predicate branches must be semantically distinct")
            )
    if len(set(candidate.parent_ids)) != len(candidate.parent_ids):
        diagnostics.append(Diagnostic("KIR908", "parent IDs must be unique"))
    if len(set(candidate.origin_proposal_ids)) != len(candidate.origin_proposal_ids):
        diagnostics.append(Diagnostic("KIR909", "origin proposal IDs must be unique"))
    return tuple(diagnostics)


def _predicate_holds(predicate: PolicyPredicate, value: int) -> bool:
    if predicate.operator is PredicateOperator.LT:
        return value < predicate.threshold
    if predicate.operator is PredicateOperator.LE:
        return value <= predicate.threshold
    if predicate.operator is PredicateOperator.EQ:
        return value == predicate.threshold
    if predicate.operator is PredicateOperator.GE:
        return value >= predicate.threshold
    if predicate.operator is PredicateOperator.GT:
        return value > predicate.threshold
    raise ValueError("unsupported predicate operator")


def evaluate_policy(
    candidate: RuntimePolicyCandidate, context: PolicyEvaluationContext
) -> int:
    """Resolve a verifier-clean policy to one prepare chunks-per-CTA value."""

    diagnostics = verify_policy(candidate)
    if diagnostics:
        raise ValueError("\n".join(str(item) for item in diagnostics))
    values = {
        PolicyFeature.TOTAL_CHUNKS: context.total_chunks,
        PolicyFeature.MAX_CHUNKS: context.max_chunks,
        PolicyFeature.NUM_SEQUENCES: context.num_sequences,
        PolicyFeature.RESIDENT_GRID_CAPACITY_CTAS: context.resident_grid_capacity_ctas,
    }
    expression = candidate.root
    while isinstance(expression, PolicyNode):
        value = values[expression.predicate.feature]
        expression = (
            expression.then_branch
            if _predicate_holds(expression.predicate, value)
            else expression.else_branch
        )
    if not isinstance(expression, PolicyLeaf):
        raise TypeError("policy did not resolve to a leaf")
    return expression.chunks_per_cta


def verify_compound_lineage(
    candidate: RuntimePolicyCandidate,
    parents: Mapping[str, RuntimePolicyCandidate],
) -> Tuple[Diagnostic, ...]:
    """Require distinct structural contributions from at least two parents."""

    diagnostics = list(verify_policy(candidate))
    parent_ids = tuple(dict.fromkeys(candidate.parent_ids))
    if len(parent_ids) < 2:
        diagnostics.append(
            Diagnostic("KIR910", "compound policy requires at least two distinct parents")
        )
        return tuple(diagnostics)

    child_subtrees = {policy_subtree_id(item) for item in _walk(candidate.root)}
    inherited_subtrees = set()
    parent_id_set = set(parent_ids)
    fragments_by_parent = {
        parent_id: tuple(
            fragment
            for fragment in candidate.lineage_fragments
            if fragment.parent_id == parent_id
        )
        for parent_id in parent_ids
    }
    if any(
        fragment.parent_id not in parent_id_set
        for fragment in candidate.lineage_fragments
    ):
        diagnostics.append(
            Diagnostic("KIR911", "lineage fragment references an undeclared parent")
        )

    for parent_id in parent_ids:
        parent = parents.get(parent_id)
        if parent is None:
            diagnostics.append(
                Diagnostic("KIR912", "compound parent does not resolve: %s" % parent_id)
            )
            continue
        if canonical_policy_id(parent) != parent_id:
            diagnostics.append(
                Diagnostic("KIR913", "compound parent ID does not match its semantics")
            )
        fragments = fragments_by_parent[parent_id]
        if not fragments:
            diagnostics.append(
                Diagnostic("KIR914", "compound parent has no inherited fragment proof")
            )
            continue
        parent_subtrees = {policy_subtree_id(item) for item in _walk(parent.root)}
        for fragment in fragments:
            if fragment.subtree_id not in parent_subtrees:
                diagnostics.append(
                    Diagnostic("KIR915", "lineage fragment is absent from its parent")
                )
            elif fragment.subtree_id not in child_subtrees:
                diagnostics.append(
                    Diagnostic("KIR916", "lineage fragment is absent from the compound child")
                )
            else:
                inherited_subtrees.add(fragment.subtree_id)

    if len(inherited_subtrees) < 2:
        diagnostics.append(
            Diagnostic("KIR917", "compound policy needs two distinct inherited subtrees")
        )
    return tuple(diagnostics)
