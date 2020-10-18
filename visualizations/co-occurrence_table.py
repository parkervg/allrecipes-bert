import sys
sys.path.append('masked_language_predict_prob/')
import pandas as pd
from collections import defaultdict
from pre_process_data import pre_process_data
"""
Creating csv with index verbs and columns nouns. Values represent the noun/verb co-occurence count.
"""

verbs, nouns, train_df = pre_process_data()

coocc_dict = defaultdict(lambda: defaultdict(int))
cooc_df = pd.DataFrame(index = verbs, columns=nouns)
for noun, verb in zip(train_df["base_noun"], train_df["base_verb"]):
    coocc_dict[verb][noun] += 1
for verb in coocc_dict:
    for noun, noun_count in coocc_dict[verb].items():
        cooc_df[noun][cooc_df.index == verb] = noun_count
cooc_df = cooc_df.fillna(0)
cooc_df.to_csv("visualizations/co-occurrence.csv")
