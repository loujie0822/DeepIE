#### 一、RUN

1. git clone https://github.com/loujie0822/DeepIE.git .
2. 安装所依赖的环境
3. 上传数据：创建data/BaiduIE_2019/或data/BaiduIE_2020/，将数据集上传到这里
4. 上传预训练模型：创建transformer_cpt/bert/，将模型相关参数上传到这里，如需要roberta则创建transformer_cpt/chinese_roberta_wwm_large_ext_pytorch/ 
5. 运行

```
export PYTHONPATH=`pwd`
python run/relation_extraction/etl_span_transformers/main.py 
  --input data/BaiduIE_2020/  
  --output finetune_model_path/ 
  --bert_model transformers_model_path 
  --max_len 128 
  --train_batch_size 96 
  --learning_rate 2e-5 
  --epoch_num 20 
```

⚠️注意，可以将 etl_span_transformers替换为其余方法（etl_span & etl_stl & multi_head_selection）进行实验。



#### **二、使用方法说明**

DeepIE中的relation_extraction共提供了四种方法

1. etl_span：来自论文——Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy，采取用pointer-net解码。

2. etl_span_transformers：在etl_span的基础上采用transformers等预训练模型。

3. etl_stl：与etl_span类似，将用pointer-net解码转化为BIES的序列标注模式（stl）

4. multi_head_selection：多头选择机制，来自论文——Joint entity recognition and relation extraction as a multi-head selection problem

   

#### 三、结果展示

- **2019语言与智能技术竞赛：关系抽取任务**

| 方法                                       | f(dev)     | p(dev)     | r(dev)     |
| ------------------------------------------ | ---------- | ---------- | ---------- |
| multi head selection                       | 76.36      | 79.24      | 73.69      |
| ETL-BIES                                   | 77.07%     | 77.13%     | 77.06%     |
| ETL-Span                                   | 78.94%     | 80.11%     | 77.8%      |
| ETL-Span + word2vec                        | 79.99%     | 80.62%     | 79.38%     |
| ETL-Span + word2vec + adversarial training | 80.38%     | 79.95%     | 80.82%     |
| ETL-Span + BERT                            | **81.88%** | **82.35%** | **81.42%** |

⚠️：ETL-Span + BERT将max_len改为256，f1达到**82.1**，使用ETL-Span + ROBERTa-large，f1为**82.6+**

TODO：将多种词向量进行拼接（bigram+word2vec+BERT）

- **2020语言与智能技术竞赛：关系抽取任务**

  (**目前只是个baseline**)

⚠️要改变原有数据的标注模式，详见code说明。

| 方法            | f(dev) | p(dev) | r(dev) |
| --------------- | ------ | ------ | ------ |
| ETL-Span + BERT | 74.58  | 74.44  | 74.71  |





