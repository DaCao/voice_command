import tensorflow as tf
import numpy as np
import pickle
import re



with open('training_v2.pickle', 'rb') as handle:
    parsed_df = pickle.load(handle)

