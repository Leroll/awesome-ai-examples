[Englist](README.md)｜[中文](README_ZH.md)

<h1 align="center">
  📜 Awesome AI Examples
</h1>

欢迎来到 Awesome AI Examples 仓库！这个仓库致力于提供高质量和实用的人工智能算法和应用示例，特别是大语言模型（LLM）、GPT、Transformers等前沿技术的具体实现。


## 💫 illustration 
- 图解位置编码 - [Sinusoidal](illustration/位置编码-Sinusoidal.ipynb)

## ⚡️ Evalution
- [C-Eval基准](https://cevalbenchmark.com/) 的本地评测代码。 [eval_ceval](evaluate/eval_ceval.py)

## 🔧 Utils
### 内存管理
- 清理显存函数 [clean_memory](utils/clean_memory.py)
- 强制释放 NVIDIA 显存 [clean_nvidia_vram.sh](utils/clean_nvidia_vram.sh)

### 文件与驱动加载
- colab 加载 Google Drive [load_google_drive](utils/load_google_drive.py)

### 训练设置
- 固定模型训练中的各种随机化操作，让模型训练过程可复现 [set_deterministic](utils/set_deterministic.py)

### 工具函数
- 计算函数运行时间的装饰器 [time_cost](utils/time_cost.py)
- 交换字典的 key-value，得到 value-key 的新字典 [reverse_dict](utils/reverse_dict.py)





