
from google.colab import drive
import os

def load_google_drive(path):
    """path 为 my google drive 内部路径 
    """

    drive.mount('/content/drive')
    root_path = '/content/drive/My Drive/'
    
    os.chdir(root_path + path)
    print(f"cwd: {os.getcwd()}")