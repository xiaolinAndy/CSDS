import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

from models.topic import MultiTopicModel
from others.vocab_wrapper import VocabWrapper
from others.id_wrapper import VocIDWrapper


class Model(nn.Module):
    def __init__(self, args, device, vocab, checkpoint=None):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.voc_id_wrapper = VocIDWrapper('pretrain_emb/id_word2vec.voc.txt')
        # Topic Model
        # Using Golve or Word2vec embedding
        if self.args.tokenize:
            self.voc_wrapper = VocabWrapper(self.args.word_emb_mode)
            self.voc_wrapper.load_emb(self.args.word_emb_path)
            self.voc_emb = torch.tensor(self.voc_wrapper.get_emb())
        else:
            self.voc_emb = torch.empty(self.vocab_size, self.args.word_emb_size)
            xavier_uniform_(self.voc_emb)
            # self.voc_emb.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
        self.topic_model = MultiTopicModel(self.voc_emb.size(0), self.voc_emb.size(-1),
                                           args.topic_num, self.voc_emb, agent=True, cust=True)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.to(device)

    def forward(self, batch):

        all_bow, customer_bow, agent_bow = \
            batch.all_bow, batch.customer_bow, batch.agent_bow
        topic_loss, _ = self.topic_model(all_bow, customer_bow, agent_bow)

        return topic_loss
