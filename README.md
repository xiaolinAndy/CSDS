# CSDS
This is the official repo for paper [CSDS: A Fine-grained Chinese Dataset for Customer Service Dialogue Summarization](https://arxiv.org/abs/2108.13139), accepted by EMNLP 2021 main conference.

## 1. Introduction

We propose a new Chinese Customer Service Dialogue Summarization dataset  (CSDS). It aims at summarizing a dialogue considering dialogue specific features. In CSDS, each dialogue has three different types of summaries: 

- **Overall summary**: The summary condensing the main information of the whole dialogue.
- **User summary**: The summary focusing on the user's main viewpoints.
- **Agent Summary**: The summary focusing on the agent's responses.

Besides, each summary are split into several segments, where each segment represent a single topic with its topic label. (A few segments may not have topic labels.) An example annotation is given as below, and if you want to see the details of how data is represented in the json file, please check the  [introduction for CSDS](utils/dataset_introduction.md).

![](utils/example.png)

## 2. Dataset Download

- Baidu NetDisk: https://pan.baidu.com/s/1KKKNuQO5af3JQuun1G3JDg 提取码：5dii
- Google Drive: https://drive.google.com/drive/folders/1IrpEUTR2ZanN0ZKmKcbaXjruTAD9JLkF?usp=sharing

## 3. Usage

### Requirements:

- python == 3.7
- pytorch == 1.6
- files2rouge == 2.1.0
- jieba == 0.42.1
- numpy == 1.19.1
- tensorboard == 2.3.0
- tensorboardx == 2.1
- cytoolz == 0.11.0

### Instruction for PGN

1. Go to the *models/PGN/* directory.
2. Download the CSDS dataset, create a new folder named *Data/* and put CSDS under the Data folder.
3. Download the [tencent embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html) and put it under the *data_utils/embeddings* folder.
4. Run the bash file *run.sh* to train and test.

### Instruction for Fast-RL/Fast-RL-mod

1. Go to the *models/Fast-RL/* or *models/Fast-RL-mod/* directory.
2. Download the CSDS dataset, create a new folder named *dataset/* and put it under the *dataset/* folder.
3. Copy the embedding file from *models/PGN/data_utils/embeddings/dialogue_embed_word* to *models/Fast-RL/data_utils/embeddings/dialogue_embed_word* or *models/Fast-RL-mod/data_utils/embeddings/dialogue_embed_word*.
4. Run the bash file *run.sh* to train and test.

### Instruction for BERT (BERTExt/BERTAbs)

1. Go to the *models/BERT/* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *data/* folder.
3. Download the pretrained BERT model, create a new folder named *bert_base_chinese/* and put it into the *bert_base_chinese/* folder.
4. Run the bash file *run.sh* to train and test.

### Instruction for TDS-SATM/TDS-SATM-mod

1. Go to the *models/TDS-SATM/* or *models/TDS-SATM-mod/* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *data/* folder.
3. Download the pretrained BERT model, create a new folder named *bert/chinese_bert/* and put it into the *bert/chinese_bert/* folder.
4. Run the bash file *run.sh* to train and test.

### Instruction for SummaRuNNer

1. Go to the *models/SummaRuNNer* directory.
2. Download the CSDS dataset, create a new folder named *data/* and put CSDS under the *dataset/* folder.
3. Copy the embedding file from *models/PGN/data_utils/embeddings/dialogue_embed_word* to *models/SummaRuNNer/data/embeddings/dialogue_embed_word* 
4. Run the bash file *run.sh* to train and test.

### Evaluation

1. We put the output of our trained models to the *results/* folder, with the overall summary, user summary and agent summary separately. If you have trained your models, you could also put the outputs into the folder.

2. Run *utils/cal_auto_metrics.py* to evaluate through automatic metrics. Pay attention to change the file names if you want to test your own output.
3. Run *utils/qa_num.py* to evaluate the QA pair matching results (Precision, Recall, F1).

## Acknowledgement

The reference code of the provided methods are:

- [PGN](https://github.com/atulkum/pointer_summarizer)
- [FAST-RL](https://github.com/ChenRocks/fast_abs_rl)
- [BERTExt/BERTAbs](https://github.com/nlpyang/PreSumm)
- [TDS+SATM](https://github.com/RowitZou/topic-dialog-summ)
- [SummaRuNNer](https://github.com/hpzhao/SummaRuNNer)

We thanks for all these researchers who have made their codes publicly available.

## Citation

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

