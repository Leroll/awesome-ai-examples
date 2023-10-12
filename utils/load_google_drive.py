

#---------------------------------
#  colab 加载 google drive
#--------------------------------- 
from google.colab import drive
import os
def load_google_drive(path='/'):
    """
    Args:
        path:  my google drive 内部路径 
    """

    drive.mount('/content/drive')
    root_path = '/content/drive/My Drive/'
    
    os.chdir(root_path + path)
    print(f"cwd: {os.getcwd()}")


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