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
    string = re.sub(r",", "", string)
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
    string = [w for w in string if w not in stopwords]
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
    dist = None
    with open(source_path, "r") as text:
        dist = FreqDist(clean_str(text.read()))
        text.close()

    top_n = dist.most_common(vocab_size)
    vocab_idx_list = {}
    frequency = []
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
    Process convert text sentence into index based on vocab_idx_list
    """
    processed_text = []
    unknow_count = 0
    sentence_count = 0
    total = 0
    with open(source_path, "r") as text:
        s = []
        for sentence in text:
            sentence = clean_str(sentence)
            sentence_count += 1
            # Add the start symbol index
            s.append(start_symbol_idx)
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
            s.append(end_symbol_idx)
            processed_text.append(s)

    print ("There are {} OOV words".format(unknow_count))
    print ("The length of processed text is {}".format(total))
    print("The number of <s> and </s> is {}".format(sentence_count))





"""
def save_index(data, path):
    with open(path, 'w') as set:
        for d in data:
            set.writelines(json.dumps(d) + "\n")
        set.close()
"""

"""
    Initialise a word embedding from given word vector

    @Param word_vec: pre trained word vector
    @Param words: list of words

    @Return d: A json object
    @Return vocabulary: A lookup table with (token, index) pairs
"""
"""

def word_vec_embedding(word_vec, vocabulary):


    print("Create Embedding from word vectors")
    knownWords.append(np.zeros((1, 300), dtype='float32'))  # word vector of <PAD>
    knownWords.append(np.random.rand(1, 300).astype('float32'))  # word vector of <EOS>
    knownWords.append(np.random.rand(1, 300).astype('float32'))  # word vector of <UNK>

    for w in words:
        try:
            # Check whether the word is in vocabulary
            index = vocabulary[w]
        except KeyError:
            # Check whether the word can be found in word vector
            try:
                vec = word_vec[w]
                vocabulary[w] = knownWordCount
                knownWords.append(np.reshape(np.array(vec, dtype='float32'), (1, 300)))
                knownWordCount += 1
            except KeyError:
                # Add unknown words to unknown tokens
                if w not in unknownTokens:
                    unknownTokens.append(w)

    print("{} words are not found in given word vectors".format(len(unknownTokens)))

    # Random initialise the unknown tokens
    for w in unknownTokens:
        vocabulary[w] = knownWordCount
        vec = np.random.rand(1, 300).astype('float32')
        knownWordCount += 1
        unknownWord.append(vec)
    return {"known": knownWords, "unkown": unknownWord}, vocabulary
"""