from __future__ import print_function
import json
import tensorflow as tf
from ModelUtils.EmbeddingLayer import EmbeddingLayer
from ModelUtils.GRU import GRU
from ModelUtils.Sampler.UniformSampler import UniformSampler
from ModelUtils.Sampler.UnigramSampler import UnigramSampler
from ModelUtils.Estimator.AlexEstimator import AlexEstimator
from ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from ModelUtils.Estimator.NegativeEstimator import NegativeEstimator