# torch Models
基于pytorch实现各种NLP相关的模型

## Models
* sentence bert  
sentence bert的实现，用于相似度匹配的fine-tune  


* bert sim  
基于bert的相似度匹配的 fine-tune 实现  


* masked language model  
  * 基础的基于bert的mlm的pytorch实现
  * 用于pattern exploiting training，基于苏建林同学 [这篇文章](https://kexue.fm/archives/8213) 


## Data processing
* Dataprocessor  
reader 和 一些相关函数。形成初步的数据集  


* Preprocessor  
预处理相关


* Masker  
mask 相关的数据处理类,把初步数据集加工为模型可用数据



## Train processing
完成模型训练任务
* 不同model的loss计算 
* 不同的lr_schedule