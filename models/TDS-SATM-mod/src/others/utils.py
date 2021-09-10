import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def test_bleu(cand, ref):
    candidate = [line.strip() for line in open(cand, encoding='utf-8')]
    reference = [line.strip() for line in open(ref, encoding='utf-8')]
    if len(reference) != len(candidate):
        raise ValueError('The number of sentences in both files do not match.')
    if len(reference) == 0:
        return 0
    score = 0.
    for i in range(len(reference)):
        gold_list = reference[i].split()
        cand_list = candidate[i].split()
        score += sentence_bleu([gold_list], cand_list, smoothing_function=SmoothingFunction().method1)
    score /= len(reference)
    return score


def test_length(cand, ref, ratio=True):
    candidate = [len(line.split()) for line in open(cand, encoding='utf-8')]
    if len(candidate) == 0:
        return 0
    if ratio:
        reference = [len(line.split()) for line in open(ref, encoding='utf-8')]
        score = sum([candidate[i] / reference[i] for i in range(len(candidate))]) / len(candidate)
    else:
        score = sum(candidate) / len(candidate)
    return score


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    if x is None:
        return None
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.contiguous()\
         .view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def test_f1(acc_num, pred_num, gold_num):
    p = acc_num / pred_num * 1.
    r = acc_num / gold_num * 1.
    if p == 0. and r == 0.:
        f1 = -1
    else:
        f1 = (2 * p * r) / (p + r)
    return f1, p, r


"""
def rouge_results_to_str(results_dict):
    if results_dict is None:
        return "No Results.\n"
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        results_dict["rouge_l_precision"] * 100
    )
"""


def rouge_results_to_str(results_dict):
    if results_dict is None:
        return "No Results.\n"
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge-1"]['f'] * 100,
        results_dict["rouge-2"]['f'] * 100,
        results_dict["rouge-l"]['f'] * 100,
        results_dict["rouge-1"]['r'] * 100,
        results_dict["rouge-2"]['r'] * 100,
        results_dict["rouge-l"]['r'] * 100,
        results_dict["rouge-1"]['p'] * 100,
        results_dict["rouge-2"]['p'] * 100,
        results_dict["rouge-l"]['p'] * 100
    )
