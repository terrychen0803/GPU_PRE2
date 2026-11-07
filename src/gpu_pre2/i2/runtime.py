"""The three shared helpers required by I2 and I3 workers.

Copied from the concrete runtime behavior in legacy E1/O1/O4 without
importing full historical packages or their datasets.
"""
from __future__ import annotations
import ctypes

def precision_policy(torch):
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        policy = "fp32_precision=ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy = "allow_tf32=False"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    return {"tf32": policy, "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "fp16_reduced_precision_reduction": False, "autocast": False}

def reset_cuda_context_for_trace(cuda_version):
    major = str(cuda_version).split(".", 1)[0]
    if not major.isdecimal():
        raise RuntimeError(f"Invalid CUDA runtime version: {cuda_version!r}")
    runtime = ctypes.CDLL(f"libcudart.so.{major}")
    reset = runtime.cudaDeviceReset
    reset.restype = ctypes.c_int
    reset.argtypes = []
    result = int(reset())
    if result != 0:
        raise RuntimeError(f"cudaDeviceReset failed with error code {result}")
    return result

def dispatch_inventory(torch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    class Inventory(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.rows=[];self.active=False;self.stage="";self.iteration=-1;self.index=0
        def begin(self,stage,index):
            self.active=True;self.stage=stage;self.iteration=index;self.index=0
        def end(self):
            self.active=False
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if not self.active:
                return func(*args,**(kwargs or {}))
            idx=self.index;self.index+=1
            label=f"o4.op/{self.stage}/{self.iteration:06d}/{idx:04d}"
            tensors=[x for x in tree_flatten((args,kwargs or {}))[0]
                     if isinstance(x,torch.Tensor)]
            metadata={"scope":label,"stage":self.stage,"iteration":self.iteration,
                      "op_index":idx,"aten":str(func),
                      "input_shapes":[list(x.shape) for x in tensors],
                      "input_dtypes":[str(x.dtype) for x in tensors]}
            torch.cuda.nvtx.range_push(label)
            try:
                out=func(*args,**(kwargs or {}))
            finally:
                torch.cuda.nvtx.range_pop()
            self.rows.append(metadata)
            return out
    return Inventory()
