from __future__ import print_function
import PredictNextWord


def main():
    print("Dealing with Large number unigram test")
    estimator_types = ["BER", "RANKH", "RANKO"]
    params = {"sampler_type": "unigram", "sample_size": 250,
              "batch_size": 20,
              "sentence_len": 70, "epoch": 10, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.001, "l_rate": 0.02, "save_step": 15000}
    distortions = [0.0, 0.75, 1.0]
    for e in estimator_types:
        for d in distortions:
            params["estimator_type"] = e
            params["distrotion"] = d
            PredictNextWord.training(params)

main()