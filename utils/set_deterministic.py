import torch
import random
import os
import numpy as np

def set_deterministic(seed=42):
    """
    让pytorch训练过程可复现，固定各种随机化的操作。
    """
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 禁止hash随机化
    os.environ['PYTHONHASHSEED'] = str(seed)

    # https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn那边，使用相同的算法，而不是经过benchmark后在几个中选取最快的。
    torch.backends.cudnn.benchmark = False

    # torch内部一些方法使用确定型的算法，具体list见函数文档
    # torch.use_deterministic_algorithms(True)

    # 当上述cudnn使用同一算法时，有可能算法本身不是确定性的，因此需要下述设定
    # 但是该设定已经被上面的设定包含了。
    torch.backends.cudnn.deterministic = True

    # dataloader在多进程时也会有reproducibility的问题
    # 这部分暂时不涉及。