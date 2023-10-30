<h1 align="center">
  torch models
</h1>

基于pytorch的:  
* 各种模型实现  
* 训练 pipeline  
* 使用工具函数  

其他各个模块位属于分离的解耦的组件，在tasks任务中选用各种组件形成完整的任务pipeline

## 🌈 Update
- 2023-10-30 新增C-eval数据机的本地评测代码
- 2023-10-27 新增图解算法模块，针对各式算法做一些可视化的理解。


## 💫 illustration 
- 图解位置编码 - [Sinusoidal](illustration/位置编码-Sinusoidal.ipynb)

## ⚡️ Evalution
- [C-Eval基准](https://cevalbenchmark.com/) 的[本地评测代码](evaluate/eval_ceval.py)


## 🌟 Models
* sentence bert  
  sentence bert的实现，用于相似度匹配的fine-tune  


* bert sim  
  基于bert的相似度匹配的 fine-tune 实现  


* masked language model  
  * 基础的基于bert的mlm的pytorch实现
  * 用于pattern exploiting training，基于苏建林同学 [这篇文章](https://kexue.fm/archives/8213) 
  * Masker  
    mask 相关的数据处理类,把初步数据集加工为模型可用数据


## 🥤 Data processing
* Dataprocessor  
  reader 和 一些相关函数。形成初步的数据集  


* Preprocessor  
  预处理相关

  
## 🔥 Train processing
完成模型训练任务
* 模型训练过程 
* 兼容不同模型的 loss，acc 计算函数
* 断点重新训练功能