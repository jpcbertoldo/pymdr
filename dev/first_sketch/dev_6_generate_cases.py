# different cases of distances on the node 'table-0' of tables-2.html

import random


CASE_TEMPLATE = """
{{
    1: {{
        ((0, 1), (1, 2)): {},
        ((1, 2), (2, 3)): {},
        ((2, 3), (3, 4)): {},
        ((3, 4), (4, 5)): {},
        ((4, 5), (5, 6)): {},
        ((5, 6), (6, 7)): {},
        ((6, 7), (7, 8)): {},
        ((7, 8), (8, 9)): {},
        ((8, 9), (9, 10)): {},
        ((9, 10), (10, 11)): {},
        ((10, 11), (11, 12)): {},
    }},
    2: {{
        ((0, 2), (2, 4)): {},
        ((2, 4), (4, 6)): {},
        ((4, 6), (6, 8)): {},
        ((6, 8), (8, 10)): {},
        ((8, 10), (10, 12)): {},
        ((1, 3), (3, 5)): {},
        ((3, 5), (5, 7)): {},
        ((5, 7), (7, 9)): {},
        ((7, 9), (9, 11)): {},
    }},
    3: {{
        ((0, 3), (3, 6)): {},
        ((3, 6), (6, 9)): {},
        ((6, 9), (9, 12)): {},

        ((1, 4), (4, 7)): {},
        ((4, 7), (7, 10)): {},

        ((2, 5), (5, 8)): {},
        ((5, 8), (8, 11)): {},
    }}
}}
"""


def generate_case_bool_sequence(length, probability, probability_cond_1):
    def bernoulli():
        return random.random() < probability

    def bernoulli_conditional(case_before_1):
        p = probability_cond_1 if case_before_1 else probability
        return random.random() < p

    sequence = [False] * length
    for i, e in enumerate(sequence):
        if i == 0:
            sequence[i] = bernoulli()
        else:
            sequence[i] = bernoulli_conditional(sequence[i - 1])

    return sequence


def generate_case():
    n_to_replace = CASE_TEMPLATE.count("{}")
    sequence = generate_case_bool_sequence(n_to_replace, 0.4, 0.7)
    sequence = ["CLOSE_ENOUGH" if val else "TOO_FAR" for val in sequence]
    case_str = CASE_TEMPLATE.format(*sequence)
    return case_str


if __name__ == "__main__":
    print(generate_case())
