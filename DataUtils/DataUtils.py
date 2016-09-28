"""
The mothods in this file are used to pre-process text data
"""
from nltk import FreqDist
import json
import numpy as np
import re


def clean_str(s):
    """
    Remove unexpected tokens from string
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    @Param s: A input string to clean
    @Return: Array of lower case word tokens
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
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
    return string.lower().split(" ")


def create_vocabulary(path, vocab_size=45000, alpha=1.0):
    """
    Create vocabulary, calculate statistics of give text data and covert text data into vocabulary index

    @Param Path: The path to target text data
    @Param vocab_size: The size of produced vocabulary default is 45,000
    @Param alpha: The power factor on empirical frequency

    @Return vocabulary: A lookup table with (token, index) pairs
    @Return frequency: The empirical frequency of each token in vocabulary
    @Return processed_text: The index form of text data, which is produced according to vocabulary
    """

    """
    Read the raw text
    """
    dist = None
    with open(path, "r") as text:
        dist = FreqDist(clean_str(text.read()))
        text.close()

    top_n = dist.most_common(vocab_size)
    vocabulary = top_n.keys()

    """
    Process text insert special token
    """
    processed_text = []
    unknow_count = 0
    sentence_count = 0
    with open(path, "r") as text:
        for sentence in text:
            sentence = clean_str(sentence)
            sentence_count += 1
            processed_text.append("<s>")
            for w in sentence:
                if w not in vocabulary:
                    processed_text.append("<unk>")
                    unknow_count += 1
            processed_text.append("</s>")

    top_n["<s>"] = sentence_count
    top_n["</s>"] = sentence_count
    top_n["<unk>"] = unknow_count

    """
    Create vocabulary and calculate frequency
    """

    vocabulary = {}
    frequency = []
    total_len = len(processed_text) + 2*sentence_count
    index = 0
    for w, f in top_n:
        vocabulary[w] = index
        index += 1
        frequency.append(np.power(f/total_len, alpha))

    """
    Convert text data into index form according to vocabulary
    """
    for i in range(total_len):
        processed_text[i] = vocabulary[processed_text[i]]

    return vocabulary, frequency, processed_text

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