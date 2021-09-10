import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim, topic_num, noise_rate=0.5):

        super(TopicModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.topic_num = topic_num
        self.noise_rate = noise_rate
        self.mlp = nn.Sequential(
            nn.Linear(vocab_size, 2*hidden_dim),
            nn.Tanh()
        )
        self.mu_linear = nn.Linear(2*hidden_dim, hidden_dim)
        self.sigma_linear = nn.Linear(2*hidden_dim, hidden_dim)
        self.theta_linear = nn.Linear(hidden_dim, topic_num)
        self.topic_emb = nn.Parameter(torch.empty(topic_num, hidden_dim))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                p.data.zero_()

    def forward(self, bow_repre, voc_emb, summ_target=None):

        id_mask = bow_repre.gt(0).float()
        # bow_valid = bow_repre.gt(0)
        # stopnum = (bow_valid.sum(dim=-1).float() * (1-self.stop_word_rate)).long()
        # threshold = bow_repre.sort(dim=-1, descending=True)[0].index_select(-1, stopnum).diagonal(0)
        # id_mask = bow_repre.gt(threshold.unsqueeze(-1)).float()
        # id_mask = bow_repre.gt(torch.mean(bow_repre[bow_repre.gt(0)])).float()
        # id_mask = bow_repre.gt(torch.medium(bow_repre, dim=-1))

        # Inference Stage
        linear_output = self.mlp(bow_repre)
        mu = self.mu_linear(linear_output)
        log_sigma_sq = self.sigma_linear(linear_output)

        eps = torch.empty_like(mu).float().normal_()
        sigma = torch.sqrt(torch.exp(log_sigma_sq))

        if self.training:
            h = mu + sigma * eps
        else:
            h = mu

        theta_logits = self.theta_linear(h)

        e_loss = -0.5 * torch.sum(1 + log_sigma_sq - mu.pow(2) - torch.exp(log_sigma_sq))

        # Generation Stage
        self.beta = beta = F.softmax(torch.matmul(self.topic_emb, voc_emb.transpose(0, 1)) / math.sqrt(self.hidden_dim), dim=-1)

        if summ_target is not None:

            summ_topic_num = int(self.topic_num * (1-self.noise_rate))
            # build noise target
            noise_target = (id_mask != summ_target).float()

            summ_mask = torch.zeros_like(theta_logits)
            summ_mask[:, summ_topic_num:] = -float('inf')

            noise_mask = torch.zeros_like(theta_logits)
            noise_mask[:, :summ_topic_num] = -float('inf')

            theta_summ = F.softmax(theta_logits + summ_mask, dim=-1)
            theta_noise = F.softmax(theta_logits + noise_mask, dim=-1)

            logits_summ = torch.log(torch.matmul(theta_summ, beta) + 1e-40)
            logits_noise = torch.log(torch.matmul(theta_noise, beta) + 1e-40)

            g_loss = - torch.sum(logits_summ * summ_target) - torch.sum(logits_noise * noise_target)
            # topic_emb = torch.cat([torch.matmul(theta_summ, self.topic_emb),
            #                        torch.matmul(theta_noise, self.topic_emb)], -1)
            topic_emb = (torch.matmul(theta_summ, self.topic_emb), torch.matmul(theta_noise, self.topic_emb))
        else:
            theta = F.softmax(theta_logits, dim=-1)
            logits = torch.log(torch.matmul(theta, beta) + 1e-40)
            g_loss = - torch.sum(logits * id_mask)
            topic_emb = torch.matmul(theta, self.topic_emb)

        return e_loss + g_loss, topic_emb


class MultiTopicModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim, topic_num, noise_rate, embeddings, agent=False, cust=False):

        super(MultiTopicModel, self).__init__()
        self.embeddings = nn.Parameter(embeddings)
        self.agent = agent
        self.cust = cust
        self.tm1 = TopicModel(vocab_size, hidden_dim, topic_num, noise_rate)
        if cust:
            self.tm2 = TopicModel(vocab_size, hidden_dim, topic_num, noise_rate)
        if agent:
            self.tm3 = TopicModel(vocab_size, hidden_dim, topic_num, noise_rate)

    def forward(self, all_bow, customer_bow, agent_bow,
                summ_all_target=None, summ_customer_target=None, summ_agent_target=None):

        loss_all, emb_all = self.tm1(all_bow, self.embeddings, summ_all_target)
        if self.cust:
            loss_customer, emb_customer = self.tm2(customer_bow, self.embeddings, summ_customer_target)
        else:
            loss_customer, emb_customer = 0, None
        if self.agent:
            loss_agent, emb_agent = self.tm3(agent_bow, self.embeddings, summ_agent_target)
        else:
            loss_agent, emb_agent = 0, None
        loss = loss_all + loss_agent + loss_customer

        return loss, (emb_all, emb_customer, emb_agent)
