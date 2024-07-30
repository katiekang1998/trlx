import json
from factscore.factscorer import FactScorer
import numpy as np
import pickle
import os

samples = np.load("../plots_base_model_samples.npy")
names = np.load("../movies/test_titles.npy")

generations = []

for i in range(2):
    for sample in samples:
        generations.append(sample[i].lstrip())

topics = np.concatenate([names, names])

import IPython; IPython.embed()


fs = FactScorer(openai_key="/data/katie_kang/openai_key_file.txt", data_dir="/data/katie_kang/trlx/examples/.cache/factscore", cache_dir="/data/katie_kang/trlx/examples/.cache/factscore")
out = fs.get_score(list(topics), list(generations), gamma=0)

import IPython; IPython.embed()


with open("../plots_base_model_samples_factscore.json", "w") as f:
    json.dump(out, f)