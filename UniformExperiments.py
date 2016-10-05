from __future__ import print_function
import PredictNextWord


def main():
    print("Dealing with Large number unigram test")
    estimator_types = ["IMP"]
    params = {"sampler_type": "unigram", "sample_size": 250,
              "batch_size": 50,
              "sentence_len": 70, "epoch_step": 100, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.001, "l_rate": 0.02, 'distortion': 0.0}

    for e in estimator_types:
         params["estimator_type"] = e
         PredictNextWord.training(params)

    estimator_types = ["ALEX", "NEG"]
    params["distortion"] = 0.75

    for e in estimator_types:
        params["estimator_type"] = e
        PredictNextWord.training(params)




main()
