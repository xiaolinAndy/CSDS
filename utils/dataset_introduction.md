# Dataset Introduction

We split CSDS into three sets: train.json, val.json, test.json.

The structure of each sample in the dataset is as:

- DialogueID: The id of each sample
- QRole: The role of user Q. Candidates include: 用户（Customer），商家（Merchant），物流师傅（Deliveryman）
- QA: Lists of Question-Answer pairs. For each pair, we have:
  - QueSumm: The user summary for this QA pair
  - AnsSummShort: The original agent summary for this QA pair, some maybe incomplete
  - AnsSummLong: The complete agent summary for this QA pair
  - QueSummUttIDs: List of key utterance ids for user summary in this QA pair
  - AnsSummShortUttIDs: List of key utterance ids for original agent summary in this QA pair
  - AnsSummLongUttIDs: List of key utterance ids for complete agent summary in this QA pair
  - QASumm: The overall summary for this QA pair
  - Topic: The topic information for this QA pair. The topic lists are given in xxx.txt. Some QAs have empty topic since the topic does not belong to our predefined list.
- Session_id: The original id in JDDC dataset
- Dialogue: The dialogue content, list of utterances, for each utterance, we have:
  - speaker: The role information for this utterance, Q stands for user and A stands for agent
  - turn: The turn index, start from 0
  - utterance: The content of this utterance. Spaces represent word split result.
- UserSumm: List of user summaries for each QA, we concatenate them to obtain the whole user summary.
- AgentSumm: List of agent summaries for each QA, we concatenate them to obtain the whole agent summary.
- FinalSumm: List of the overall summaries for each QA, we concatenate them to obtain the whole overall summary.

