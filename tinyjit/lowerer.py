from llvmlite import ir

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
            self._lower_elementwise_binary(op, self.builder.fadd)
        elif op.op_type == "sub":
            self._lower_elementwise_binary(op, self.builder.fsub)
        elif op.op_type == "mul":
            self._lower_elementwise_binary(op, self.builder.fmul)
        elif op.op_type == "div":
            self._lower_elementwise_binary(op, self.builder.fdiv)
        elif op.op_type == "pow":
            self._lower_pow(op)
        elif op.op_type == "exp":
            self._lower_elementwise_unary(op, "llvm.exp")
        elif op.op_type == "log":
            self._lower_elementwise_unary(op, "llvm.log")
        elif op.op_type == "sqrt":
            self._lower_elementwise_unary(op, "llvm.sqrt")
        elif op.op_type == "sin":
            self._lower_elementwise_unary(op, "llvm.sin")
        elif op.op_type == "cos":
            self._lower_elementwise_unary(op, "llvm.cos")
        elif op.op_type == "abs":
            self._lower_elementwise_unary(op, "llvm.fabs")
        elif op.op_type == "tan":
            self._lower_tan(op)
        elif op.op_type == "dot":
            self._lower_vector_dot_product(op)
        elif op.op_type == "norm":
            self._lower_norm(op)
        elif op.op_type == "determinant":
            self._lower_determinant(op)
        elif op.op_type == "add_scalar":
            self._lower_scalar_binary(op, self.builder.fadd)
        elif op.op_type == "matmul": 
            self.lower_mul(op)
        elif op.op_type == "derivative":
            self._lower_derivative(op)
        
        else: 
            raise NotImplementedError(f"Op {op.op_type} not supported yet.")
    

    #generic imp for elementwise binary ops 
    def _lower_elementwise_binary(self, op, binary_op_func): 
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
        loop_cond_block= self.llvm_func.append_basic_block(f"{op.op_type}loop_cond")
        loop_body_block= self.llvm_func.append_basic_block(f"{op.op_type}loop_body")
        loop_end_block= self.llvm_func.append_basic_block(f"{op.op_type}loop_end")
        
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
        c_val = self.builder.fadd(a_val, b_val, name=f"{op.op_type}c_val")
        
        #store to C[i]
        c_ptr = self.builder.gep(C_ptr, [i_val], name="c_ptr")
        self.builder.store(c_val, c_ptr)

        #i += 1
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond_block)
        self.builder.position_at_start(loop_end_block) 
    
    #generic imp for elementwise unary op
    def _lower_elementwise_unary(self, op, intrinsic_name): 
        A, = op.inputs
        B, = op.outputs

        A_ptr = self.tensor_map[A.name]
        B_ptr = self.tensor_map[B.name]

        size = 1 
        for dim in B.shape:
            size *= dim 

        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)
        # Basic loop blocks
        loop_cond_block = self.llvm_func.append_basic_block(f"{op.op_type}_loop_cond")
        loop_body_block = self.llvm_func.append_basic_block(f"{op.op_type}_loop_body")
        loop_end_block = self.llvm_func.append_basic_block(f"{op.op_type}_loop_end")
        
        # Branch from current to condition block
        self.builder.branch(loop_cond_block)
        
        # Loop condition: i < size
        self.builder.position_at_start(loop_cond_block)
        i_val = self.builder.load(i_ptr, name="i_val")
        cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), size), name="cond")
        self.builder.cbranch(cond, loop_body_block, loop_end_block)
        
        # Loop body
        self.builder.position_at_start(loop_body_block)
        
        # Load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")
        
        # Call the appropriate intrinsic
        intrinsic = self.module.declare_intrinsic(intrinsic_name, [ir.FloatType()])
        b_val = self.builder.call(intrinsic, [a_val], name=f"{op.op_type}_result")
        
        # Store to B[i]
        b_ptr = self.builder.gep(B_ptr, [i_val], name="b_ptr")
        self.builder.store(b_val, b_ptr)
        
        # i = i + 1
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond_block)
        
        # End loop
        self.builder.position_at_start(loop_end_block)

    def _lower_tan(self, op): #tan(x) = sin(x) / cos(x)

        A, = op.inputs
        B, = op.outputs
        
        A_ptr = self.tensor_map[A.name]
        B_ptr = self.tensor_map[B.name]
        
        # Calculate total size
        size = 1
        for dim in B.shape:
            size *= dim
        
        # Allocate loop counter
        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)
        
        # Basic loop blocks
        loop_cond_block = self.llvm_func.append_basic_block("tan_loop_cond")
        loop_body_block = self.llvm_func.append_basic_block("tan_loop_body")
        loop_end_block = self.llvm_func.append_basic_block("tan_loop_end")
        
        # Branch from current to condition block
        self.builder.branch(loop_cond_block)
        
        # Loop condition: i < size
        self.builder.position_at_start(loop_cond_block)
        i_val = self.builder.load(i_ptr, name="i_val")
        cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), size), name="cond")
        self.builder.cbranch(cond, loop_body_block, loop_end_block)
        
        # Loop body
        self.builder.position_at_start(loop_body_block)
        
        # Load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")
        
        # Calculate sin(x)
        sin_intrinsic = self.module.declare_intrinsic("llvm.sin", [ir.FloatType()])
        sin_val = self.builder.call(sin_intrinsic, [a_val], name="sin_val")
        
        # Calculate cos(x)
        cos_intrinsic = self.module.declare_intrinsic("llvm.cos", [ir.FloatType()])
        cos_val = self.builder.call(cos_intrinsic, [a_val], name="cos_val")
        
        # Calculate tan(x) = sin(x) / cos(x)
        tan_val = self.builder.fdiv(sin_val, cos_val, name="tan_val")
        
        # Store to B[i]
        b_ptr = self.builder.gep(B_ptr, [i_val], name="b_ptr")
        self.builder.store(tan_val, b_ptr)
        
        # i = i + 1
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond_block)
        
        # End loop
        self.builder.position_at_start(loop_end_block)
    
    def _lower_norm(self, op): # v = (x, y) -> ||v|| = sqrt(x**2 + y**2)

        A, = op.inputs
        B, = op.outputs
        
        A_ptr = self.tensor_map[A.name]
        B_ptr = self.tensor_map[B.name]
        
        # Initialize sum of squares to 0
        sum_sq_ptr = self.builder.alloca(ir.FloatType(), name="sum_sq")
        self.builder.store(ir.Constant(ir.FloatType(), 0.0), sum_sq_ptr)
        
        # Get vector length
        vec_len = A.shape[0]
        
        # Loop over vector elements
        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)
        
        loop_cond = self.llvm_func.append_basic_block("norm_loop_cond")
        loop_body = self.llvm_func.append_basic_block("norm_loop_body")
        loop_end = self.llvm_func.append_basic_block("norm_loop_end")
        
        self.builder.branch(loop_cond)
        
        # Loop condition: i < vec_len
        self.builder.position_at_start(loop_cond)
        i_val = self.builder.load(i_ptr, name="i_val")
        cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), vec_len), name="norm_cond")
        self.builder.cbranch(cond, loop_body, loop_end)
        
        # Loop body: sum_sq += A[i] * A[i]
        self.builder.position_at_start(loop_body)
        
        # Load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")
        
        # Square
        a_sq = self.builder.fmul(a_val, a_val, name="a_sq")
        
        # Accumulate
        sum_sq = self.builder.load(sum_sq_ptr, name="sum_sq")
        sum_sq = self.builder.fadd(sum_sq, a_sq, name="sum_sq_updated")
        self.builder.store(sum_sq, sum_sq_ptr)
        
        # Increment i
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_cond)
        
        # End loop and calculate square root
        self.builder.position_at_start(loop_end)
        sum_sq = self.builder.load(sum_sq_ptr, name="final_sum_sq")
        
        # Calculate square root
        sqrt_intrinsic = self.module.declare_intrinsic("llvm.sqrt", [ir.FloatType()])
        norm_val = self.builder.call(sqrt_intrinsic, [sum_sq], name="norm_val")
        
        # Store result
        self.builder.store(norm_val, B_ptr)
    
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
            self._lower_vec_mat_mul(A_ptr, B_ptr, C_ptr, A.shape[0], B.shape[0])
        #mat-vec mul 
        elif len(B.shape) == 1: 
            self._lower_mat_vec_mul(A_ptr, B_ptr, C_ptr, A.shape[0], A.shape[1])

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
    
    def _lower_vec_mat_mul(self, A_ptr, B_ptr, C_ptr, k, n): 
        """Vector-matrix mul (k,) @ (k,n) -> (n,)"""

        #init all elements of C to 0 
        for j in range(n):
            c_ptr = self.builder.gep(C_ptr, [ir.Constant(ir.IntType(32), j)], name=f"c_init_ptr{j}")
            self.builder.store(ir.Constant(ir.FloatType(), 0.0), c_ptr)

        #-------
        # for j < n; j++ ......
        #      for i <k; i++......
        #          instructions....
        #      end loop i 
        # end loop j
        #-------------
        #init i, j iterator
        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        j_ptr = self.builder.alloca(ir.IntType(32), name="j")

        # for each col j 
        self.builder.store(ir.Constant(ir.IntType(32), 0), j_ptr)

        loop_j_cond = self.llvm_func.append_basic_block("loop_j_cond")
        loop_j_body = self.llvm_func.append_basic_block("loop_j_body")
        loop_j_end = self.llvm_func.append_basic_block("loop_j_end")

        self.builder.branch(loop_j_cond)

        #j loop cond 
        self.builder.position_at_start(loop_j_cond)
        j_val = self.builder.load(j_ptr, name="j_val")
        j_cond = self.builder.icmp_signed("<", j_val, ir.Constant(ir.IntType(32), n), name="j_cond")
        self.builder.cbranch(j_cond, loop_j_body, loop_j_end)

        #j loop body 
        self.builder.position_at_start(loop_j_body)

        #i = 0 (inner loop)
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)        
     
        loop_i_cond = self.llvm_func.append_basic_block("loop_i_cond")
        loop_i_body = self.llvm_func.append_basic_block("loop_i_body")
        loop_i_end = self.llvm_func.append_basic_block("loop_i_end")

        #i loop cond 
        self.builder.position_at_start(loop_i_cond)
        i_val = self.builder.load(i_ptr, name="i_val")
        i_cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), k), name="i_cond")
        self.builder.cbranch(i_cond, loop_i_body, loop_i_end)

        #i loop body C[i] += A[i] * B[i, j]
        self.builder.position_at_start(loop_i_body)

        #load A[i]
        a_ptr = self.builder.gep(A_ptr, [i_val], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")

        # Load B[i,j] = B[i*n + j]
        b_idx = self.builder.mul(i_val, ir.Constant(ir.IntType(32), n), name="b_row_offset")
        b_idx = self.builder.add(b_idx, j_val, name="b_idx")
        b_ptr = self.builder.gep(B_ptr, [b_idx], name="b_ptr")
        b_val = self.builder.load(b_ptr, name="b_val")

        #mul 
        prod = self.builder.fmul(a_val, b_val, name="prod")

        #Store to C[j]
        c_ptr = self.builder.gep(C_ptr, [j_val], name="c_ptr")
        c_val = self.builder.load(c_ptr, name="c_val")
        c_val = self.builder.fadd(c_val, prod, name="c_val_updated")
        self.builder.store(c_val, c_ptr)
    
        #i++
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_i_cond)
        
        # End i loop
        self.builder.position_at_start(loop_i_end)
        
        #j++
        j_next = self.builder.add(j_val, ir.Constant(ir.IntType(32), 1), name="j_next")
        self.builder.store(j_next, j_ptr)
        self.builder.branch(loop_j_cond)
        
        # End j loop
        self.builder.position_at_start(loop_j_end)

    def _lower_mat_vec_mul(self, A_ptr, B_ptr, C_ptr, m, k): 
        """mat-vec mul: (m, k) @ (k,) -> (m,)"""

        for i in range(m):
            c_ptr = self.builder.gep(C_ptr, [ir.Constant(ir.IntType(32), i)], name=f"c_init_ptr_{i}")
            self.builder.store(ir.Constant(ir.FloatType(), 0.0), c_ptr)

        #-------
        # for i < m; i++ ......
        #      for j <k; j++......
        #          instructions....
        #      end loop j
        # end loop i
        #-------------
        i_ptr = self.builder.alloca(ir.IntType(32), name="i")
        j_ptr = self.builder.alloca(ir.IntType(32), name="j")
        
        # For each row i
        self.builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)
        
        loop_i_cond = self.llvm_func.append_basic_block("loop_i_cond")
        loop_i_body = self.llvm_func.append_basic_block("loop_i_body")
        loop_i_end = self.llvm_func.append_basic_block("loop_i_end")
        
        self.builder.branch(loop_i_cond)
        
        # i loop condition
        self.builder.position_at_start(loop_i_cond)
        i_val = self.builder.load(i_ptr, name="i_val")
        i_cond = self.builder.icmp_signed("<", i_val, ir.Constant(ir.IntType(32), m), name="i_cond")
        self.builder.cbranch(i_cond, loop_i_body, loop_i_end)
        
        # i loop body
        self.builder.position_at_start(loop_i_body)
        
        # Reset j to 0 for inner loop
        self.builder.store(ir.Constant(ir.IntType(32), 0), j_ptr)
        
        loop_j_cond = self.llvm_func.append_basic_block("loop_j_cond")
        loop_j_body = self.llvm_func.append_basic_block("loop_j_body")
        loop_j_end = self.llvm_func.append_basic_block("loop_j_end")
        
        self.builder.branch(loop_j_cond)
        
        # j loop condition
        self.builder.position_at_start(loop_j_cond)
        j_val = self.builder.load(j_ptr, name="j_val")
        j_cond = self.builder.icmp_signed("<", j_val, ir.Constant(ir.IntType(32), k), name="j_cond")
        self.builder.cbranch(j_cond, loop_j_body, loop_j_end)
        
        # j loop body: C[i] += A[i,j] * B[j]
        self.builder.position_at_start(loop_j_body)
        
        # Load A[i,j] = A[i*k + j]
        a_idx = self.builder.mul(i_val, ir.Constant(ir.IntType(32), k), name="a_row_offset")
        a_idx = self.builder.add(a_idx, j_val, name="a_idx")
        a_ptr = self.builder.gep(A_ptr, [a_idx], name="a_ptr")
        a_val = self.builder.load(a_ptr, name="a_val")
        
        # Load B[j]
        b_ptr = self.builder.gep(B_ptr, [j_val], name="b_ptr")
        b_val = self.builder.load(b_ptr, name="b_val")
        
        # mul
        prod = self.builder.fmul(a_val, b_val, name="prod")
        
        # store to C[i]
        c_ptr = self.builder.gep(C_ptr, [i_val], name="c_ptr")
        c_val = self.builder.load(c_ptr, name="c_val")
        c_val = self.builder.fadd(c_val, prod, name="c_val_updated")
        self.builder.store(c_val, c_ptr)
        
        #j++
        j_next = self.builder.add(j_val, ir.Constant(ir.IntType(32), 1), name="j_next")
        self.builder.store(j_next, j_ptr)
        self.builder.branch(loop_j_cond)
        
        #j++
        self.builder.position_at_start(loop_j_end)
        
        #i++
        i_next = self.builder.add(i_val, ir.Constant(ir.IntType(32), 1), name="i_next")
        self.builder.store(i_next, i_ptr)
        self.builder.branch(loop_i_cond)
        
        # End i loop
        self.builder.position_at_start(loop_i_end)

