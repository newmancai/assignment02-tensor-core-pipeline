#!/usr/bin/env python3
"""Exact-rational CPU probes for KDA time/depth semantics; no dependencies.

This is a minimal mixer-only KDA, not a trained Transformer or a GPU test.
q = k = 1, alpha = 1, beta = 1/2, v = previous-depth hidden.
Residuals and MLPs are omitted because one valid counterexample suffices to
disprove general state-sharing equivalence. Fractions remove rounding ambiguity.
"""

from fractions import Fraction as F
import json


def step(value, state):
    decayed = state  # alpha = 1
    updated = decayed + F(1, 2) * (value - decayed)
    return updated, updated  # q = 1, so output equals scalar state


def depth_major(values, loops=2, carry_final_between_loops=False):
    hidden = list(map(F, values))
    final = F(0)
    all_outputs = []
    for _ in range(loops):
        state = final if carry_final_between_loops else F(0)
        output = []
        for value in hidden:
            result, state = step(value, state)
            output.append(result)
        final = state
        hidden = output
        all_outputs.append(output)
    return all_outputs


def token_major(values, loops=2, share_one_state=False):
    states = [F(0)] * (1 if share_one_state else loops)
    all_outputs = [[] for _ in range(loops)]
    for value in map(F, values):
        hidden = value
        for depth in range(loops):
            state_index = 0 if share_one_state else depth
            hidden, states[state_index] = step(hidden, states[state_index])
            all_outputs[depth].append(hidden)
    return all_outputs


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def determinant(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def transition(key, beta, decay):
    delta = [[F(i == j) - beta * key[i] * key[j]
              for j in range(2)] for i in range(2)]
    diagonal = [[decay[i] if i == j else F(0)
                 for j in range(2)] for i in range(2)]
    return matmul(delta, diagonal), matmul(diagonal, delta)


def as_json(value):
    if isinstance(value, F):
        return {"fraction": str(value), "float": float(value)}
    raise TypeError(type(value).__name__)


def main():
    original = [1, 2]
    changed_future = [1, 4]
    correct_full = depth_major(original)
    correct_stream = token_major(original)
    correct_changed = depth_major(changed_future)
    leaky = depth_major(original, carry_final_between_loops=True)
    leaky_changed = depth_major(changed_future, carry_final_between_loops=True)
    flattened = token_major(original, share_one_state=True)
    assert correct_full == correct_stream
    assert correct_full[-1][0] == correct_changed[-1][0] == F(1, 4)
    assert leaky[-1][0] == F(7, 8)
    assert leaky_changed[-1][0] == F(11, 8)
    assert flattened != correct_full

    # Unit vectors with rational coordinates avoid approximate normalization.
    key = [F(3, 5), F(4, 5)]
    second_key = [F(4, 5), F(-3, 5)]
    assert sum(x * x for x in key) == 1
    decay = [F(1, 2), F(1)]
    kda_transition_matrix, reversed_order = transition(key, F(1), decay)
    assert kda_transition_matrix != reversed_order

    beta_rows = []
    for beta in [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)]:
        a, _ = transition(key, beta, decay)
        det = determinant(a)
        identity = (1 - beta) * decay[0] * decay[1]
        assert det == identity and det >= 0
        beta_rows.append({"beta": beta, "determinant": det})
    a1, _ = transition(key, F(1, 2), decay)
    a2, _ = transition(second_key, F(3, 4), [F(1), F(3, 4)])
    product_det = determinant(matmul(a2, a1))
    assert product_det == determinant(a2) * determinant(a1) == F(3, 64)
    reflection, _ = transition(key, F(2), [F(1), F(1)])
    assert determinant(reflection) == -1

    print(json.dumps({
        "status": "PASS",
        "arithmetic": "exact fractions; CPU standard library only",
        "causality": {
            "input": original,
            "changed_future_input": changed_future,
            "correct_depth_major": correct_full,
            "correct_token_major": correct_stream,
            "correct_first_output_after_future_change": correct_changed[-1][0],
            "wrong_carry_sequence_final": leaky,
            "wrong_first_output_after_future_change": leaky_changed[-1][0],
            "wrong_shared_state_token_major": flattened,
        },
        "noncommutativity": {
            "key": key,
            "decay": decay,
            "beta": F(1),
            "kda_A_equals_delta_times_D": kda_transition_matrix,
            "wrong_A_equals_D_times_delta": reversed_order,
        },
        "determinant_domain": {
            "beta_in_unit_interval": beta_rows,
            "two_factor_product_determinant": product_det,
            "beta_two_reflection_determinant": determinant(reflection),
            "limitation": "transition matrices only, not full nonlinear models",
        },
    }, default=as_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
