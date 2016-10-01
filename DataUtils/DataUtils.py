"""
The mothods in this file are used to pre-process text data
"""
from nltk import FreqDist
import json
import numpy as np
import nltk
import re


def clean_str(s):
    """
    Remove unexpected tokens from string
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    @Param s: A input string to clean
    @Return: Array of lower case word tokens
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower().split(" ")
    stopwords = nltk.corpus.stopwords.words('english')
    string = [w for w in string if w not in stopwords and not '']
    return string


def create_vocabulary(source_path, vocab_size=40000):
    """
    Create vocabulary, calculate statistics of give text data and covert text data into vocabulary index

    @Param Path: The path to target text data
    @Param vocab_size: The size of produced vocabulary default is 40,000
    @Param alpha: The power factor on empirical frequency

    @Return vocabulary: A lookup table with (token, index) pairs
    @Return frequency: The empirical frequency of each token in vocabulary
    @Return processed_text: The index form of text data, which is produced according to vocabulary
    """

    """
    Create the vocabulary from raw text
    """
    words = []
    iters = 0
    sentences = []
    with open(source_path, "r") as text:
        for line in text:
            iters += 1
            if iters % 100000 is 0:
                print("processed {} sentence".format(iters))
                break
            sentence = clean_str(line)
            sentence.pop()
            sentences.append(sentence)
            words += sentence
        text.close()

    dist = FreqDist(words)
    top_n = dist.most_common(vocab_size)
    vocab_idx_list = {}
    frequency = [0.0]
    for i in range(len(top_n)):
        w, f = top_n[i]
        vocab_idx_list[w] = i+1
        frequency.append(f)

    """
    Insert special token into vocabulary
    """
    start_symbol_idx = len(vocab_idx_list) + 1
    end_symbol_idx = start_symbol_idx + 1
    unk_symbol_idx = end_symbol_idx + 1
    vocab_idx_list["<s>"] = start_symbol_idx
    vocab_idx_list["</s>"] = end_symbol_idx
    vocab_idx_list["<unk>"] = unk_symbol_idx
    vocab_idx_list["<pad>"] = 0

    """
    Convert text sentence into index based on vocab_idx_list
    """
    processed_text = []
    unknow_count = 0
    sentence_count = 0
    total = 0
    unk_word_list = []
    for sentence in sentences:
        s = [start_symbol_idx]
        idx = unk_symbol_idx
        for w in sentence:
            total += 1
            idx = unk_symbol_idx
            # check whether the word w is in the vocabulary
            try:
                # if w in the vocabulary then add the idx of w
                idx = vocab_idx_list[w]
                s.append(idx)
            except KeyError:
                # if w is not in then insert the unk_symbol_idx
                s.append(idx)
                unknow_count += 1
                if unknow_count < 5000:
                    unk_word_list.append(w)
        s.append(end_symbol_idx)
        processed_text.append(s)

    print ("There are {} OOV words".format(unknow_count))
    print ("The length of processed text is {}".format(total))
    print("The number of <s> and </s> is {}".format(sentence_count))

    for i in range(len(frequency)):
        frequency[i] /= float(total)

    frequency.append(float(sentence_count)/float(total))
    frequency.append(float(sentence_count) / float(total))
    frequency.append(float(unknow_count) / float(total))

    with open('../ProcessedData/frequency_100000.txt', 'w') as freq:
        freq.write(json.dumps(frequency))
        freq.close()
    with open('../ProcessedData/vocabulary_100000.txt', 'w') as vocab:
        vocab.write(json.dumps(vocab_idx_list))
        vocab.close()
    with open('../ProcessedData/sentences_100000.txt', 'w') as sentence:
        for s in processed_text:
            sentence.write(json.dumps(s)+"\n")
        sentence.close()