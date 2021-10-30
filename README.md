# CSDS
This is the official repo for paper [CSDS: A Fine-grained Chinese Dataset for Customer Service Dialogue Summarization](https://arxiv.org/abs/2108.13139), accepted by EMNLP 2021 main conference.

## Update

### 1. Evaluation

In the paper, we use the files2rouge package to run ROUGE scores by transferring all the Chinese characters into indexes. However, this process may cause effect to ROUGE-L score, since each summary will be recognized as one single sentence. Thus we modify the script by adding a special character after each sentence in the summary and set it to be the sentence splitting sign for files2rouge. We have already change the code in *utils/cal_auto_metrics.py*. And we present the new result as below.

### 2. Method

For CSDS, we found that the tri-gram blocking strategy does great harm, since many summaries actually have repeated tri-grams or four-grams, such as "用户说", 用户表示". Thus we modify the tri-gram blocking strategy in BERTAbs and TDS+SATM and report the new scores as below.

|             | ROUGE-1           | ROUGE-2           | ROUGE-L           | BLEU              | BS                | MS                |
| ----------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Longest     | 30.02/35.42/25.94 | 15.52/20.26/13.84 | 28.00/33.49/24.01 | 11.19/13.14/9.94  | 63.61/67.92/62.89 | 12.38/16.46/10.71 |
| LexPageRank | 36.32/35.15/30.81 | 19.43/19.29/16.56 | 34.67/33.82/29.37 | 13.48/14.14/12.65 | 66.60/67.23/65.27 | 15.01/13.94/12.26 |
| SummaRuNNer | 44.91/43.90/40.40 | 27.99/26.46/25.26 | 42.97/41.89/38.38 | 21.60/19.35/20.69 | 71.77/72.16/70.94 | 24.10/22.16/20.41 |
| BERTExt     | 43.55/37.25/35.75 | 27.51/21.58/23.05 | 41.75/35.69/34.25 | 21.59/14.91/17.39 | 71.24/68.01/67.59 | 22.69/16.06/14.59 |
| PGN         | 55.58/53.55/50.20 | 39.19/37.06/35.12 | 53.46/51.05/47.59 | 32.31/29.64/28.25 | 78.40/78.68/76.13 | 28.58/26.68/25.13 |
| Fast-RL     | 57.95/57.33/53.07 | 41.39/40.43/37.59 | 55.99/55.17/50.76 | 33.04/33.39/30.44 | 79.57/80.29/77.72 | 29.78/28.55/27.18 |
| Fast-RL*    | 57.70/58.40/52.83 | 41.24/41.68/37.38 | 55.76/56.11/50.54 | 32.94/33.53/30.11 | 79.76/81.06/77.52 | 30.12/29.95/26.89 |
| BERTAbs     | 55.41/52.71/49.61 | 39.42/36.39/33.88 | 53.41/50.45/46.88 | 27.77/30.17/27.02 | 79.23/79.23/76.41 | 28.11/24.95/23.91 |
| TDS+SATM    | 51.69/54.20/49.16 | 34.94/36.70/33.15 | 49.44/51.66/46.35 | 22.89/25.82/26.22 | 77.47/79.21/76.06 | 25.35/26.13/24.19 |
| TDS+SATM*   | 53.14/53.82/47.37 | 35.98/36.64/31.55 | 50.68/51.56/44.65 | 26.47/25.47/22.72 | 77.81/79.29/75.52 | 26.11/26.12/23.09 |

## Instructions

### 1. Introduction

We propose a new Chinese Customer Service Dialogue Summarization dataset  (CSDS). It aims at summarizing a dialogue considering dialogue specific features. In CSDS, each dialogue has three different types of summaries: 

- **Overall summary**: The summary condensing the main information of the whole dialogue.
- **User summary**: The summary focusing on the user's main viewpoints.
- **Agent Summary**: The summary focusing on the agent's responses.

Besides, each summary are split into several segments, where each segment represent a single topic with its topic label. (A few segments may not have topic labels.) An example annotation is given as below, and if you want to see the details of how data is represented in the json file, please check the  [introduction for CSDS](utils/dataset_introduction.md).

![](utils/example.png)

### 2. Dataset Download

- Baidu NetDisk: https://pan.baidu.com/s/1KKKNuQO5af3JQuun1G3JDg 提取码：5dii
- Google Drive: https://drive.google.com/drive/folders/1IrpEUTR2ZanN0ZKmKcbaXjruTAD9JLkF?usp=sharing

### 3. Usage

#### Requirements:

- python == 3.7
- pytorch == 1.6
- files2rouge == 2.1.0
- jieba == 0.42.1
- numpy == 1.19.1
- tensorboard == 2.3.0
- tensorboardx == 2.1
- cytoolz == 0.11.0

#### Instruction for PGN

1. Go to the *models/PGN/* directory.
2. Download the CSDS dataset, create a new folder named *Data/* and put CSDS under the Data folder.
3. Download the [tencent embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html) and put it under the *data_utils/embeddings* folder.
4. Run the bash file *run.sh* to train and test.

#### Instruction for Fast-RL/Fast-RL-mod

1. Go to the *models/Fast-RL/* or *models/Fast-RL-mod/* directory.
2. Download the CSDS dataset, create a new folder named *dataset/* and put it under the *dataset/* folder.
3. Copy the embedding file from *models/PGN/data_utils/embeddings/dialogue_embed_word* to *models/Fast-RL/data_utils/embeddings/dialogue_embed_word* or *models/Fast-RL-mod/data_utils/embeddings/dialogue_embed_word*.
4. Run the bash file *run.sh* to train and test.

#### Instruction for BERT (BERTExt/BERTAbs)

1. Go to the *models/BERT/* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *data/* folder.
3. Download the pretrained BERT model, create a new folder named *bert_base_chinese/* and put it into the *bert_base_chinese/* folder.
4. Run the bash file *run.sh* to train and test.

#### Instruction for TDS-SATM/TDS-SATM-mod

1. Go to the *models/TDS-SATM/* or *models/TDS-SATM-mod/* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *data/* folder.
3. Download the pretrained BERT model, create a new folder named *bert/chinese_bert/* and put it into the *bert/chinese_bert/* folder.
4. Run the bash file *run.sh* to train and test.

#### Instruction for SummaRuNNer

1. Go to the *models/SummaRuNNer* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *dataset/* folder.
3. Copy the embedding file from *models/PGN/data_utils/embeddings/dialogue_embed_word* to *models/SummaRuNNer/data/embeddings/dialogue_embed_word* 
4. Run the bash file *run.sh* to train and test.

#### Evaluation

1. We put the output of our trained models to the *results/* folder, with the overall summary, user summary and agent summary separately. If you have trained your models, you could also put the outputs into the folder.

2. Run *utils/cal_auto_metrics.py* to evaluate through automatic metrics. Pay attention to change the file names if you want to test your own output.
3. Run *utils/qa_num.py* to evaluate the QA pair matching results (Precision, Recall, F1).

### Acknowledgement

The reference code of the provided methods are:

- [PGN](https://github.com/atulkum/pointer_summarizer)
- [FAST-RL](https://github.com/ChenRocks/fast_abs_rl)
- [BERTExt/BERTAbs](https://github.com/nlpyang/PreSumm)
- [TDS+SATM](https://github.com/RowitZou/topic-dialog-summ)
- [SummaRuNNer](https://github.com/hpzhao/SummaRuNNer)

We thanks for all these researchers who have made their codes publicly available.

### Citation

If you want to cite our paper, please use this temporarily:

```
@misc{lin2021csds,
      title={CSDS: A Fine-Grained Chinese Dataset for Customer Service Dialogue Summarization}, 
      author={Haitao Lin and Liqun Ma and Junnan Zhu and Lu Xiang and Yu Zhou and Jiajun Zhang and Chengqing Zong},
      year={2021},
      eprint={2108.13139},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

The EMNLP proceeding version will be given after the official publication.

If you have any issues, please contact with haitao.lin@nlpr.ia.ac.cn

