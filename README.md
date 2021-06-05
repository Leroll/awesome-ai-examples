持续测试各种无监督的文本表示方法

---
## Models
1. word2vec 
2. bert pool
3. bert hidden average
   

4. 权重
    1. tf-idf
    2. smooth inverse frequency
    
---
## Result 
|Dataset  |Algorithm | F1-score|
|:--------|:----------|:--------|
|bp_corpus|word2vec||
|bp_corpus|bert_pool||
|bp_corpus|bert_avg||
|bp_corpus|tf-idf + word2vec||
|bp_corpus|tf-idf + bert_pool||
|bp_corpus|tf-idf + bert_avg||
|bp_corpus|sif + word2vec||
|bp_corpus|sif + bert_pool||
|bp_corpus|sif + bert_avg||