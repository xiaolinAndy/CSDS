import sys
import os
import hashlib
import subprocess
import collections

import json
import tarfile
import io
import pickle as pkl
import re
import nltk


dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"

# These are the number of .story files we expect there to be in cnn_stories_dir
# and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

def get_speaker(context):
    speakers = set()
    for sent in context:
        sent = sent.split()
        if len(sent) > 0:
            speakers.add(sent[0])
    return list(speakers)


def tokenize_stories(stories, add_speaker):
    total_count, avg_sum_len, avg_context_len, avg_sum_sent, avg_context_sent = 0, 0., 0., 0., 0.
    speaker_count = {}
    with open(stories, 'r') as f:
        data = json.load(f)
    processed_data = []
    for sample in data:
        total_count += 1
        sum = sample['summary']
        sum = sum.lower()
        sum = nltk.word_tokenize(sum)
        avg_sum_len += len(sum)
        sum_sents = []
        while len(sum) > 0:
            try:
                fst_period_idx = sum.index(".")
            except ValueError:
                fst_period_idx = len(sum)
            sent = sum[:fst_period_idx + 1]
            sum = sum[fst_period_idx + 1:]
            sum_sents.append(' '.join(sent))
        avg_sum_sent += len(sum_sents)
        #sum = ['<s> ' + sent + ' </s>' for sent in sum]

        context = sample['dialogue']
        id = sample['id']
        context = context.lower()
        context = re.sub('\r\n', '\n', context)
        context = re.split('\n', context)
        avg_context_sent += len(context)
        context = [fix_missing_period(s) for s in context]
        context = [nltk.word_tokenize(s) for s in context]
        for c in context:
            avg_context_len += len(c)
        context = [' '.join(s) for s in context]
        if len(context) > 1:
            if add_speaker:
                final_context = []
                speakers = get_speaker(context)
                if len(speakers) > 10:
                    print(context, sum_sents, speakers)
                    exit()
                if len(speakers) not in speaker_count.keys():
                    speaker_count[len(speakers)] = 1
                else:
                    speaker_count[len(speakers)] += 1
                for sent in context:
                    if sent.strip() == '':
                        continue
                    sent += ' |'
                    for s in speakers:
                        if sent[:len(s)] != s:
                            sent += ' ' + s
                    final_context.append(sent)
                processed_data.append([sum_sents, final_context])
            else:
                processed_data.append([sum_sents, context])

    print(processed_data[0])
    print('total count: ', total_count)
    print('avg sum len: ', avg_sum_len/total_count)
    print('avg sum sent: ', avg_sum_sent/total_count)
    print('avg context len: ', avg_context_len/total_count)
    print('avg context sent: ', avg_context_sent/total_count)
    print(speaker_count)

    return processed_data


def read_story_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 2 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n\n')
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(story_file):
    """ return as list of sentences"""
    lines = read_story_file(story_file)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.lower().strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def write_to_tar(data, out_file, makevocab=False):

    if makevocab:
        vocab_counter = collections.Counter()

    with tarfile.open(out_file, 'w') as writer:
        for idx, (abstract, article) in enumerate(data):
            if idx % 1000 == 0:
                print("Writing story {} of {}; {:.2f} percent done".format(
                    idx, len(data), float(idx)*100.0/float(len(data))))

            # Write to JSON file
            js_example = {}
            js_example['id'] = idx
            js_example['article'] = article
            js_example['abstract'] = abstract
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = ' '.join(article).split()
                abs_tokens = ' '.join(abstract).split()
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)
    print(len(vocab_counter))

    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join('final_data_add/', "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory {} contains {} files"
            " but should contain {}".format(
                stories_dir, num_stories, num_expected)
        )


if __name__ == '__main__':
    add_speaker = True
    train_set = tokenize_stories('train.json', add_speaker)
    val_set = tokenize_stories('val.json', add_speaker)
    test_set = tokenize_stories('test.json', add_speaker)

    # Read the tokenized stories, do a little postprocessing
    # then write to bin files
    #write_to_tar(test_set, os.path.join('final_data_add/', "test.tar"))
    #write_to_tar(val_set, os.path.join('final_data_add/', "val.tar"))
    #write_to_tar(train_set, os.path.join('final_data_add/', "train.tar"),
                 #makevocab=True)