# torch Models

---
基于pytorch实现各种NLP相关的模型

以下是相关文档介绍

## Data processing

---
|class|备注|
|---|---|
|Dataprocessor|reader 和 一些相关函数。形成初步的数据集|
|Preprocessor|预处理相关|
|Masker|mask 相关的数据处理类|

## Models

---
|model|备注|
|---|---|
|sentence bert fine-tune|实现sentence bert|
|bert fine-tune|基础的bert fine-tune|
|masked language model|实现基于bert的MLM|
|pattern exploiting training|pet的一个变体，苏建林同学 [这篇文章](https://kexue.fm/archives/8213) 的pytorch 实现。|