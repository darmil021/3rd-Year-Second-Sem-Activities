# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:26:07 2024

@author: Dariel Militante
"""

import numpy as np
import matplotlib.pyplot as plt

prior_probs = np.array([[0.33, 0.3], [0.2, 0.17]])

plt.imshow(prior_probs, cmap='gray')
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.annotate(prior_probs[i, j], (j, i), color="red", fontsize=20, fontweight='bold', ha='center', va='center')

plt.title('Prior Probabilities', fontsize=20)

# calculate the probability of cancer patient and diagnostic test

# calculate P(A|B) given P(A), P(B|A), P(B|not A)
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    # calculate P(not A)
    not_a = 1 - p_a

    # calculate P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a

    # calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

# P(A)
p_a = 0.0002

# P(B|A)
p_b_given_a = 0.85

# P(not A)
p_b_given_not_a = 0.05

# calculate P(A|B)
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)

# summarize
print('P(A|B) = %.3f%%' % (result * 100))