#!/usr/bin/env python3
"""
Debug script to understand why the l-solver is failing.
"""

import numpy as np

def objective(l, gamma, p):
    """Objective function for the Kelly growth equation."""
    if l <= 1 or p <= 0 or p >= 1:
        return 1e9
    term1 = np.log(l)
    term2 = p * np.log(p)
    term3 = (1 - p) * np.log((1 - p) / (l - 1))
    return term1 + term2 + term3 - gamma

# Test parameters
known_p = 0.7
known_l = 2.5

# Calculate theoretical gamma
term1 = np.log(known_l)
term2 = known_p * np.log(known_p)
term3 = (1 - known_p) * np.log((1 - known_p) / (known_l - 1))
theoretical_gamma = term1 + term2 + term3

print(f"Test Parameters: p = {known_p}, l = {known_l}")
print(f"Theoretical gamma: {theoretical_gamma:.6f}")
print()

# Test the objective function across different l values
print("Testing objective function across l values:")
print("l\t\tf(l)")
print("-" * 30)

l_values = [1.01, 1.1, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 50.0, 100.0]
for l in l_values:
    f_val = objective(l, theoretical_gamma, known_p)
    print(f"{l:.2f}\t\t{f_val:.6f}")

print()
print("Looking for where f(l) changes sign...")
print()

# Find where the function changes sign
for i in range(len(l_values) - 1):
    l1, l2 = l_values[i], l_values[i+1]
    f1 = objective(l1, theoretical_gamma, known_p)
    f2 = objective(l2, theoretical_gamma, known_p)
    
    if f1 * f2 < 0:  # Different signs
        print(f"Sign change between l={l1:.2f} (f={f1:.6f}) and l={l2:.2f} (f={f2:.6f})")
        print(f"This means the root is in the interval [{l1:.2f}, {l2:.2f}]")
    else:
        print(f"No sign change between l={l1:.2f} (f={f1:.6f}) and l={l2:.2f} (f={f2:.6f})")

print()
print("Let's also check what happens when l approaches 1:")
for l in [1.001, 1.01, 1.1, 1.2]:
    f_val = objective(l, theoretical_gamma, known_p)
    print(f"l={l:.3f}: f(l)={f_val:.6f}") 