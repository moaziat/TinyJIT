import ctypes
import numpy as np
import llvmlite.binding as llvm 
from tinyjit.ir import Lowerer

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


class JIT: 

    def __init__(self):
        target = llvm.Target.from_default_triple()
        self.target_machine = target.create_target_machine()

        self.module = None 
        self.engine = None
        self._keep_alive = []
    def compile(self, llvm_ir: str, func_name: str): 
        #Parse the IR
        self.module = llvm.parse_assembly(llvm_ir)
        self.module.verify()

        #create execution engine 
        self.engine = llvm.create_mcjit_compiler(self.module, self.target_machine)
        self.engine.finalize_object()

        #get function ptr 
        func_ptr = self.engine.get_function_address(func_name)
        return func_ptr
    
    def __del__(self): 
        #clean to prevent mem leaks 
        if self.engine:
            del self.engine
        if self.module:
            del self.module 

#-- Use JIT as a functon
def jit(func_ir):
    # Lower IR to LLVM 
    llvm_ir = Lowerer(func_ir).lower()

    #create JIT compiler 
    jit_compiler = JIT()
    #create a reference to the compiler
    _jit_ref = jit_compiler
    try: 
        fn_ptr = jit_compiler.compile(str(llvm_ir), func_ir.name)
    except Exception as e: 
        print(f"Compilation failed: {e}")
        raise

    def wrapped(*args): 
        if len(args) != len(func_ir.args): 
            raise ValueError("Incorrect numbers of arguments passed to JIT")
        
        input_arrays = []
        for arg in args:
            if not isinstance(arg, np.ndarray):
                arg = np.array(arg, dtype=np.float32)
            elif arg.dtype != np.float32:
                arg = arg.astype(np.float32)
            input_arrays.append(arg)
        
        output_arrays = []
        for out_tensor in func_ir.outputs:
            out_array = np.zeros(out_tensor.shape, dtype=np.float32)
            output_arrays.append(out_array)
        
        #Prepare C arguments (pointers)
        c_args = []
        for arr in args: 
            c_args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        for arr in output_arrays: 
            c_args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        argtypes = [ctypes.POINTER(ctypes.c_float)] * len(c_args)
        cfunc = ctypes.CFUNCTYPE(None, *argtypes)(fn_ptr)

        cfunc(*c_args)

        if len(output_arrays) == 1: 
            return output_arrays[0]
        return tuple(output_arrays)
    #ref to jit compiler
    wrapped._jit_compiler = jit_compiler

    return wrapped