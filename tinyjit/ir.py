from llvmlite import ir

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
    
    def __add__(self, other): 
        assert self.shape == other.shape, "Shape mismatch for matrix addition"
        out_tensor = Tensor(self.shape, dtype=self.dtype)
        op = Op("add", inputs=[self, other], outputs=[out_tensor])
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
    def __init__(self, op_type, inputs, outputs): 
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = f"op{Op._counter}"
        Op._counter += 1 

    def __repr__(self): 
        in_names = ", ".join(t.name for t in self.inputs)
        out_names = ", ".join(t.name for t in self.outputs)
        return f"{self.name}: {out_names} = {self.op_type}({in_names})"


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
    

class Lowerer: 

    def __init__(self, function): 
        self.function = function 
        self.module = ir.Module(name="tinyjit_module")
        self.builder= None
        self.llvm_func= None
        self.tensor_map = {}

    def lower(self):
        return_type = ir.VoidType() 
        arg_types = []
        for _ in self.function.args: 
            arg_types.append(ir.PointerType(ir.FloatType()))
        for _ in self.function.outputs: 
             arg_types.append(ir.PointerType(ir.FloatType()))
            
        func_type= ir.FunctionType(return_type, arg_types)
        self.llvm_func = ir.Function(self.module, func_type, name=self.function.name)

        #map all tensors to func args (inputs + outputs)
        all_tensors = list(self.function.args) + list (self.function.outputs)
        #Map IR tensors to LLVM function arguments
        for ir_arg, llvm_arg in zip(all_tensors, self.llvm_func.args): 
            llvm_arg.name = ir_arg.name
            self.tensor_map[ir_arg.name] = llvm_arg

        entry_block  = self.llvm_func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_block)

        for op in self.function.ops:
            self.lower_op(op)

        self.builder.ret_void()        

        return self.module
    
    def lower_op(self, op):
        if op.op_type == "add": 
            self.lower_add(op)
        elif op.op_type == "matmul": 
            self.lower_mul(op)
        else: 
            raise NotImplementedError(f"Op {op.op_type} not supported yet.")
    def lower_add(self, op): 
        A, B = op.inputs
        C, = op.outputs

        A_ptr = self.tensor_map[A.name]
        B_ptr = self.tensor_map[B.name]
        C_ptr = self.tensor_map[C.name]

        size = 1 
        for dim in C.shape: 
            size *= dim
        
        #allocate loop counter i = 0
        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)

        # Basic loop blocks
        loop_cond_block= self.llvm_func.append_basic_block("loop_cond")
        loop_body_block= self.llvm_func.append_basic_block("loop_body")
        loop_end_block= self.llvm_func.append_basic_block("loop_end")
        
        #Branch from entry to condition block
        self.builder.branch(loop_cond_block)

        self.builder.position_at_start(loop_cond_block)
        i_val = self.builder.load(i_ptr, name="i_val")
        cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), size), name="cond")
        self.builder.cbranch(cond, loop_body_block, loop_end_block)

        self.builder.position_at_start(loop_body_block)

        #load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")

        #load B[i]
        b_ptr = self.builder.gep(B_ptr, [i_val], name="b_ptr")
        b_val = self.builder.load(b_ptr, name="b_val")

        #A[i] + B[i]
        c_val = self.builder.fadd(a_val, b_val, name="c_val")
        
        #store to C[i]
        c_ptr = self.builder.gep(C_ptr, [i_val], name="c_ptr")
        self.builder.store(c_val, c_ptr)

        #i += 1
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond_block)
        self.builder.position_at_start(loop_end_block) 
    
    def lower_mul(self, op): 
        A, B = op.inputs
        C, = op.outputs

        A_ptr = self.tensor_map[A.name]
        B_ptr = self.tensor_map[B.name]
        C_ptr = self.tensor_map[C.name]

        #vec-vec mul
        if len(A.shape) == 1 and len(B.shape) == 1: 
            self._lower_vector_dot_product(A_ptr, B_ptr, C_ptr, A.shape[0])
        #vec-mat mul 
        elif len(A.shape) == 1: 
            self._lower_vec_mat_mul(A_ptr, B_ptr, A.shape[0], B.shape[0])


    def _lower_vector_dot_product(self, A_ptr, B_ptr, C_ptr, k): 
        """ Vector dot product (k,) @ (k,) -> (1,)"""
        
        #init res to 0
        self.builder.store(ir.Constant(ir.FloatType(), 0.0), C_ptr)

        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)

        loop_cond = self.llvm_func.append_basic_block("dot_loop_cond")
        loop_body = self.llvm_func.append_basic_block("dot_loop_body")
        loop_end = self.llvm_func.append_basic_block("dot_loop_end")

        self.builder.branch(loop_cond)

        #loop condition: i < k
        self.builder.position_at_start(loop_cond)
        i_val = self.builder.load(i_ptr, name="i_val")
        cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), k), name="dot_cond")
        self.builder.cbranch(cond, loop_body, loop_end)

        #loop body: res += A[i] * B[i]
        self.builder.position_at_start(loop_body)

        # Load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")
        
        # Load B[i]
        b_ptr = self.builder.gep(B_ptr, [i_val], name="b_ptr")
        b_val = self.builder.load(b_ptr, name="b_val")

        #mul 
        prod = self.builder.fmul(a_val, b_val, name="prod")

        #Store to C[i]
        c_val = self.builder.load(C_ptr, name="c_val")
        c_val = self.builder.fadd(c_val, prod, name="c_val_updated")
        self.builder.store(c_val, C_ptr)

        #increment i 
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond)

        #end loop 
        self.builder.position_at_start(loop_end)