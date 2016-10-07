from __future__ import print_function
import PredictNextWord


def main():
    print("Dealing with Large number unigram test")

    estimator_types = ["BLA", "NEG", "BER", "ALEX", "IMP"]
    params = {"sampler_type": "unigram", "sample_size": 500,
              "batch_size": 100,
              "sentence_len": 70, "epoch": 35, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.001, "l_rate": 0.02, 'distortion': 1.0, 'save_step': 15000}
    for e in estimator_types:
        params["estimator_type"] = e
        PredictNextWord.training(params)

main()
