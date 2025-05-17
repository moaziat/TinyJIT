import ctypes
import numpy as np
import llvmlite.binding as llvm 
from tinyjit.ir import Lowerer

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


class JIT: 

    def __init__(self):
        triple = llvm.get_default_triple()
        target = llvm.Target.from_triple(triple)
        self.target_machine = target.create_target_machine()

        self.backing_mod = llvm.parse_assembly("")
        self.engine = llvm.create_mcjit_compiler(self.backing_mod, self.target_machine)

    def compile(self, llvm_ir: str, func_name: str): 
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        func_ptr = self.engine.get_function_address(func_name)
        return func_ptr

#-- Use JIT as a functon
def jit(func_ir):
    llvm_ir = Lowerer(func_ir).lower()
    engine = JIT()
    fn_ptr = engine.compile(str(llvm_ir), func_ir.name)

    def wrapped(*args): 
        if len(args) != len(func_ir.args): 
            raise ValueError("Incorrect numbers of arguments passed to JIT")

        c_args = []
        for arr in args: 
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_args.append(ptr)
        
        #allocate output 
        shape = args[0].shape
        out = np.zeros(shape, dtype=np.float32)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_args.append(out_ptr)

        argtypes = [ctypes.POINTER(ctypes.c_float)] * len(c_args)
        cfunc = ctypes.CFUNCTYPE(None, *argtypes)(fn_ptr)

        # Call compiled function
        cfunc(*c_args)
        return out

    return wrapped