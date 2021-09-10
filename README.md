# CSDS
This is the official repo for paper "CSDS: A Fine-grained Chinese Dataset for Customer Service Dialogue Summarization", accepted by EMNLP 2021.

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

1. Go to one of the model directory.
2. Download the CSDS dataset, create a new folder named data and put it under the data folder.
3. download the [tencent embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html) and put it under the data_utils/embedding folder.
4. Run the bash file "run.sh" to train and test.

### Instruction for Fast-RL

1. Go to one of the model directory.
2. Download the CSDS dataset, create a new folder named data and put it under the data folder.
3. download the [tencent embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html) and put it under the data_utils/embedding folder.
4. Run the bash file "run.sh" to train and test.
