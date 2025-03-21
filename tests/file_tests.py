import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from goph420_lab03.root_find_func import newton_raphson_root

def Fx(x):
    return x**2 -1
def dFx(x):
    return 2*x

newton_raphson_root(1, Fx, dFx)
# Call the function
result, iter, error = newton_raphson_root(1, Fx, dFx)
print("="*40)
print(f"Root: {result:.6}")
print(f"Iteration: {iter}")
print(f"Rel. Error: {error}")
print("="*40)

