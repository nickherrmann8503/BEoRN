from scipy.optimize import brentq



def f(x):
    return x**3-5


print(brentq(f, -5, 30))  # Should print a root of f(x) = 0 in the interval [1, 3]