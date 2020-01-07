# DeepIE: Information Extraction Toolkit based Deep Learning 
**DeepIE : 基于深度学习的信息抽取工具包**

主要内容包括：

- 实体抽取/事件段落抽取/事件主体抽取
- 实体链接
- 实体-关系抽取
- 事件抽取

涉及的数据领域包含：

- 医疗
- 金融
- 电商
- 法律

## Installation

```shell
$ git clone https://github.com/loujie0822/DeepIE
$ cd DeepIE
```

## Content 主要目录





## 主要算法介绍

- 实体抽取/事件段落抽取/事件主体抽取
  - 
- 实体链接
  - 
- 关系抽取
  - 
- 实体-关系抽取
  - 
- 事件抽取
  - 
- 评论观点抽取
  - 

## Performance

| 任务类别 | 任务数据 | 核心策略 | dev  | test |
| -------- | -------- | -------- | ---- | ---- |
| 实体抽取 |          |          |      |      |
|          |          |          |      |      |
|          |          |          |      |      |



## TODO计划

- [ ] 1月7日-10日，基本框架设计
- [ ] 1月7日-10日，add 实体抽取/实体-关系联合抽取代码

## 开发要点

- 深度学习模型的可复用部分要单独摘出。
  - 特别地，对于BERT等预训练语言模型不要复用
- 对于同一类型任务的数据源要单独处理，统一定义数据输入的scheme。
- 数据集要特别检查是否可以开源。





## Reference

1. https://github.com/zjunlp/deepke