from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from largeNC import PredictNextWord
import argparse


def main():
    print("Dealing with Large number")
    parser = argparse.ArgumentParser()
    parser.add_argument("estimator_name",
                        help="Name of the estimator one of: NEG, IMP, BLA, BER, ALEX",
                        type=str)
    parser.add_argument("folder",
                        help="Folder where to load/save parameters and results",
                        type=str)
    parser.add_argument("-ss", "--sample_size",
                        help="How many samples to take from Q",
                        type=int,
                        default=250)
    parser.add_argument("-bs", "--batch_size",
                        help="Size of the batch",
                        type=int,
                        default=100)
    parser.add_argument("-e", "--epochs",
                        help="Size of the batch",
                        type=int,
                        default=10)
    parser.add_argument("--max_len",
                        help="Maximum length of a sentence accepted",
                        type=int,
                        default=70)
    parser.add_argument("-ed", "--embed_dim",
                        help="Dimensionality of the word embeddings",
                        type=int,
                        default=100)
    parser.add_argument("-gd", "--gru_dim",
                        help="Dimensionality of the hidden state in the GRU",
                        type=int,
                        default=128)
    parser.add_argument("--lamb",
                        help="L2 Penalty",
                        type=float,
                        default=0.0)
    parser.add_argument("-lrg", "--l_rate_gru",
                        help="Learning rate for the parameters of the GRU",
                        type=float,
                        default=0.02)
    parser.add_argument("-lre", "--l_rate_embed",
                        help="Learning rate for the parameters of the embeddings",
                        type=float,
                        default=0.02)
    parser.add_argument("-d", "--distortion",
                        help="Distortion applied to the Unigram frequencies",
                        type=float,
                        default=1.0)
    parser.add_argument("-r", "--record",
                        help="Period on which to record the true LL",
                        type=int,
                        default=100)
    args = parser.parse_args()
    v = vars(args)
    print("Program arguments:")
    for name, value in v.items():
        print(name, "-", value)
    PredictNextWord.training(**v)

if __name__ == "__main__":
    main()
