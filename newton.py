# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
def derivative(f, x):
    h = 1e-5
    return (f(x + h) - f(x)) / h

def second_derivative(f, x):
    return derivative(lambda z: derivative(f, z), x)

def optimize(x0, f):
    tolerance = 1e-6
    max_iterations = 100
    x = x0
    for i in range(max_iterations):
        d1 = derivative(f, x)
        d2 = second_derivative(f, x)
        x_new = x - d1 / d2
        if abs(x_new - x) < tolerance:
            break
        x = x_new
    return x
