from tinyjit.ir import Tensor, Op, Function
from tinyjit.jit import jit
import numpy as np

# Build IR
A = Tensor((4,))
B = Tensor((4,))
C = Tensor((4,))
op = Op("add", inputs=[A, B], outputs=[C])
func = Function("addition", inputs=[A, B], outputs=[C])

# Compile
compiled = jit(func)

# Run
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
out = compiled(a, b)

print(out)