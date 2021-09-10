import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"
#tag_to_ix = {"B": 1, "I": 2, "O": 0, START_TAG: 3, STOP_TAG: 4}

def to_scalar(var):  # var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):  # vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()维度是1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢

def log_sum_exp_batch(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

class CRFLayer(nn.Module):
    def __init__(self, tag_to_ix):
        super(CRFLayer, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.transitions.data[tag_to_ix['B'], tag_to_ix['B']] = -10000
        self.transitions.data[tag_to_ix['B'], tag_to_ix['I']] = -10000
        self.transitions.data[tag_to_ix['I'], tag_to_ix['O']] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        bs = feats.shape[0]
        init_alphas = torch.Tensor(bs, self.tagset_size).fill_(-10000.).cuda() # 1*5 而且全是-10000

        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.  # 因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，

        # Wrap in a variable so that we will get automatic backprop
        forward_var = torch.tensor(init_alphas)  # 初始状态的forward_var，随着step t变化
        convert_feats = feats.permute(1, 0, 2)

        # Iterate through the sentence 会迭代feats的行数次，
        for feat in convert_feats:  # feat的维度是５ 依次把每一行取出来~
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):  # next tag 就是简单 i，从0到len
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(bs, -1).expand(bs,
                                                               self.tagset_size)  # 维度是1*5 噢噢！原来，LSTM后的那个矩阵，就被当做是emit score了

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).repeat(bs, 1)  # 维度是１＊５
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp_batch(next_tag_var).view(1, -1))
            # 此时的alphas t 是一个长度为5，例如<class 'list'>: [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t, dim=0).permute(1, 0)  # 到第（t-1）step时５个标签的各自分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[
            STOP_TAG]].view(1, -1).repeat(bs, 1)  # 最后只将最后一个单词的forward var与转移 stop tag的概率相加 tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        alpha = log_sum_exp_batch(terminal_var)  # alpha是一个0维的tensor

        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence #feats
        totalsocre_list = []
        for feat, tag in zip(feats, tags):
            totalscore = torch.zeros(1, dtype=torch.float, device=tags.device)
            tag = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=tags.device), tag])
            for i, smallfeat in enumerate(feat):
                totalscore = totalscore + \
                             self.transitions[tag[i + 1], tag[i]] + smallfeat[tag[i + 1]]
            totalscore = totalscore + self.transitions[self.tag_to_ix[STOP_TAG], tag[-1]]
            totalsocre_list.append(totalscore)
        return torch.cat(totalsocre_list)


    def neg_log_likelihood(self, feature, tags):
        forward_score = self._forward_alg(feature)  # 0维的一个得分，20.*来着
        gold_score = self._score_sentence(feature, tags)  # tensor([ 4.5836])
        return torch.sum(forward_score - gold_score)

    def _viterbi_decode(self, feats_list):
        path_list = []
        for feats in feats_list:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).to(feats.device)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_list.append(best_path)
        return path_list

    def forward_test(self, feats):  # dont confuse this with _forward_alg above.

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(feats)
        return tag_seq


# Check predictions after training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# print(model(precheck_sent)[0])  # 得分
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# print(model(precheck_sent)[1])  # tag seque