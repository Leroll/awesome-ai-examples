#---------------------------------
#  Function to clean RAM & vRAM
#--------------------------------- 
import gc
import ctypes
import torch
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()