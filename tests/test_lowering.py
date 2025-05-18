from tinyjit.ir import Tensor, Op, Function, DataType
from tinyjit.jit import jit
import numpy as np

def test_addition():
    print("Testing tensor addition...")
    
    # Create input and output tensors
    A = Tensor((4,), name="A")
    B = Tensor((4,), name="B") 
    C = A + B  # This creates the op and output tensor
    
    # Create function
    func = Function("addition", inputs=[A, B], outputs=[C])
    
    # Print IR representation
    print("IR Representation:")
    print(func)
    print()
    
    # Compile the function
    compiled_func = jit(func)
    
    # Create numpy arrays for testing
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    
    # Execute the compiled function
    result = compiled_func(a, b)
    
    # Print results
    print("Input A:", a)
    print("Input B:", b)
    print("Output C (A + B):", result)
    
    # Verify correctness
    expected = a + b
    print("Correct?", np.allclose(result, expected))

if __name__ == "__main__":
    test_addition()