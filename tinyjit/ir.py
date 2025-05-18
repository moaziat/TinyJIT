
class DataType: 
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32" 
    INT64 = "int64"
    BOOL = "bool"

class Tensor: 
    """
    Multi-dimentional array where each tensor has 
    shape: a tuple, size of each dimension 
    dtype: the data type
    op: the operation that produced to this tensor (None if it's an input tensor)
    name: a unique name for the tensor
    """
    _counter = 0
    def __init__(self, shape, dtype=DataType.FLOAT32, op=None, name=None): 
        if not all(isinstance(d, int) for d in shape):
            raise TypeError(f"Shape must be tuple of ints, got {shape}")
        if name is None: 
            name = f"t{Tensor._counter}"
            Tensor._counter += 1
        self.name = name
        self.shape = shape 
        self.dtype = dtype 
        self.op = op
    

    
    def __add__(self, other): 
        assert self.shape == other.shape, "Shape mismatch for matrix addition"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("add", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op
        return out_tensor
    
    def __sub__(self, other): 
        assert self.shape == other.shape, "Shape mismatch for matrix substraction"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("sub", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op 
        return out_tensor
    
    def __mul__(self, other): 
        assert self.shape == other.shape, "shape mismatch for multiplication"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("mul", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op 
        return out_tensor
    
    def __truediv__(self, other):
        assert self.shape == other.shape, "Shape mismatch for division"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("div", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op
        return out_tensor
        
    def __pow__(self, other):
        assert self.shape == other.shape, "Shape mismatch for power operation"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("pow", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op
        return out_tensor
    
    #unary functions
    def exp(self):
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("exp", inputs=[self], outputs=[out_tensor])
        out_tensor.op = op 
        return out_tensor
    
    def log(self):
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("log", inputs=[self], outputs=[out_tensor])
        out_tensor.op = op 
        return out_tensor
    
    def __add_scalar__(self, scalar_value): 
        out_tensor = Tensor(shape=self.shape, dtype=self.dtype)
        op = Op("add_scalar", inputs=[self], outputs=[out_tensor], attributes={"value": scalar_value})
        out_tensor.op = op 
        return out_tensor
    def __radd__(self, other):  # tensor + scalar and scalar + tensor
        if isinstance(other, (int, float)): 
            return self.__add_scalar__(other)
        return NotImplemented
    

    def __matmul__(self, other): 
        assert len(self.shape) >= 1 and len(other.shape) >= 1, "Tensors must have at least 1 dimension"
        assert self.shape[-1] == other.shape[0], "Shape mismatch for matrix multiplication"
        
        if len(self.shape) == 1 and len(other.shape) == 1: 
            #vec-vec mul: (n,) @ (n,) -> scalar (1,)
            out_shape = (1,)
        elif len(self.sahpe) == 1: 
            #vec-mat mul: (n,) @ (n,m) -> (m,)
            out_shape = other.shape[1:]
        elif len(other.shape) == 1: 
            #mat-vec mul: (m, n) @ (m,) -> (m,)
            out_shape = self.shape[:-1]
        else: 
            #ordinary matmul 
            out_shape = self.shape[:-1] + other.shape[1:]

        out_tensor = Tensor(out_shape, dtype=self.dtype)
        op = Op("matmul", inputs=[self, other], outputs=[out_tensor])
        out_tensor.op = op
        return out_tensor
    def __repr__(self): 
        return f"%{self.name}: {self.dtype}{self.shape}"
    

class Op: 
    """
    This class represents a node in the computational graph
    
    For C = A x B, we create a node represnting the matrix multiplication
    an operation node connecting input nodes A, B to output node C
    similarily for E = C + D, then we construct a directed acyclic graph

    A       B       D 
     \      /       \
      \ matmul       \      Nodes = {A, B, C, D, E}
       \    /         \     Edges = operations connect nodes
        \  C           add  Directions: from inputs -> intermidiate (C in our case) -> ouptut
         \/             /
         E <------------
    """
    _counter = 0
    def __init__(self, op_type, inputs, outputs, attributes=None): 
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}
        self.name = f"op{Op._counter}"
        Op._counter += 1 

    def __repr__(self): 
        in_names = ", ".join(t.name for t in self.inputs)
        out_names = ", ".join(t.name for t in self.outputs)
        if self.attributes:
            attr_str = " attrs={" + ", " .join(f"{k}={v}" for k, v in self.attributes.items())+ "}"
        return f"{self.name}: {out_names} = {self.op_type}({in_names}){attr_str}" 


class Function: 
    def __init__(self, name, inputs, outputs): 
        self.name = name
        self.args = inputs
        self.outputs = outputs
        self.ops = self.__collect_ops(outputs)

    def __collect_ops(self, outputs): 

        ops = []
        visited = set()

        def visit(tensor): 
            if tensor.op and tensor.op not in visited:
                for inp in tensor.op.inputs: 
                    visit(inp)
                visited.add(tensor.op)
                ops.append(tensor.op)
        
        for out in outputs:
            visit(out)
        return ops
    
    def __repr__(self):
        lines = [f"func @{self.name}(" + ", ".join(repr(inp) for inp in self.args) + ") {"]
        for op in self.ops:
            lines.append(f"  {op}")
        lines.append("  return " + ", ".join(f"%{out.name}" for out in self.outputs))
        lines.append("}")
        return "\n".join(lines)
    

