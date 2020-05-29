- **各主流方法在主要中文NER数据集上的表现情况**

|                | **lexicon** | **Ontonotes** | **MSRA**  | **Resume** | **Weibo** |
| -------------- | ----------- | ------------- | --------- | ---------- | --------- |
| biLSTM         | ----        | 71.81         | 91.87     | 94.41      | 56.75     |
| Lattice  LSTM  | 词表1       | 73.88         | 93.18     | 94.46      | 58.79     |
| WC-LSTM        | 词表1       | 74.43         | 93.36     | 94.96      | 49.86     |
| LR-CNN         | 词表1       | 74.45         | 93.71     | 95.11      | 59.92     |
| CGN            | 词表2       | 74.79         | 93.47     | 94.12      | 63.09     |
| LGN            | 词表1       | 74.85         | 93.63     | 95.41      | 60.15     |
| Simple-Lexicon | 词表1       | 75.54         | 93.50     | **95.59**  | 61.24     |
| FLAT           | 词表1       | **76.45**     | 94.12     | 95.45      | 60.32     |
| FLAT           | 词表2       | 75.70         | **94.35** | 94.93      | **63.42** |
| BERT           | ----        | 80.14         | 94.95     | 95.53      | 68.20     |
| BERT+FLAT      | 词表1       | **81.82**     | **96.09** | **95.86**  | **68.55** |

- 说明：
  - 词表1为Lattice  LSTM  采用的词表
  - 词表2为[论文](Analogical Reasoning on Chinese Morphological and Semantic Relations)采用的词表

- 具体方法对应的文献为：

  [1] [Lattice LSTM：Chinese NER Using Lattice LSTM](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.02023)

  [2] [LR-CNN:CNN-Based Chinese NER with Lexicon Rethinking](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/1698/d96c6fffee9ec969e07a58bab62cb4836614.pdf)

  [3] [CGN:Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1396.pdf)

  [4] [LGN: A Lexicon-Based Graph Neural Network for Chinese NER](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1096.pdf)

  [5] [FLAT: Chinese NER Using Flat-Lattice Transformer](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2004.11795.pdf)

  [6] [WC-LSTM: An Encoding Strategy Based Word-Character LSTM for Chinese NER Lattice LSTM](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/43d7/4cd04fb22bbe61d650861766528e369e08cc.pdf%3F_ga%3D2.158312058.1142019791.1590478401-1756505226.1584453795)

  [7] [Multi-digraph: A Neural Multi-digraph Model for Chinese NER with Gazetteers](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1141.pdf)

  [8] [Simple-Lexicon：Simplify the Usage of Lexicon in Chinese NER](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1908.05969.pdf)

