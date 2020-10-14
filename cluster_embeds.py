from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
import tensorflow as tf
import pandas as pd
import sys
from collections import Counter
import operator
from analyze_embeddings import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import regex as re
from custom_logging import Blogger
logger = Blogger()

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
model = TFBertForMaskedLM.from_pretrained("models/allrecipes/checkpoint-3500", from_pt = True, config = config)

"""
Transforming noun_id and verb_id to class_keys
TODO:
    Instead of pd.loc, look into join function
"""
train_df = pd.read_csv("example.csv")

noun_classes_df = pd.read_csv("../epic-kitchens-55-annotations/EPIC_noun_classes.csv")
noun_classes = train_df.noun_class.tolist()
nouns = [noun_classes_df.loc[noun_classes_df["noun_id"] == i, "class_key"].iloc[0] for i in noun_classes]
train_df["base_noun"] = nouns

verb_classes_df = pd.read_csv("../epic-kitchens-55-annotations/EPIC_verb_classes.csv")
verb_classes = train_df.verb_class.tolist()
verbs = [verb_classes_df.loc[verb_classes_df["verb_id"] == i, "class_key"].iloc[0] for i in verb_classes]
train_df["base_verb"] = verbs

def rearrange_colons(text):
    text = text.split(":")
    if len(text) > 1:
        text = text[1] + " " + text[0]
    else:
        text = text[0]
    return text

# Structure of cleaned_text:original_text
nouns = {rearrange_colons(x): x for x in list(set(nouns))}
verbs = {rearrange_colons(x) for x in list(set(verbs))}

# Transforming to embeddings
# verb_embeddings = convert_to_embeddings(verbs, model = model)
noun_embeddings = convert_to_embeddings(list(nouns.keys()), model = model)
X = np.array(list(noun_embeddings.values()))

def get_top_epic_verbs(noun: str, df: pd.DataFrame, k = 5, log = False):
    existing_verbs = df.loc[df["base_noun"] == noun, "base_verb"].tolist()
    top_verbs = Counter(existing_verbs)
    if log:
        logger.yellow("Top EpicKitchen verbs for '{}':".format(noun))
        for verb, count in top_verbs.most_common()[:k]:
            logger.log("      " + verb)
        print()
    return [item[0] for item in top_verbs.most_common()[:k]]


test_range = 20
results_dict = {}
for n_clusters in range(2, test_range):
    kmeans_labels = KMeans(n_clusters = n_clusters).fit_predict(X)
    for i in range(n_clusters):
        cluster_nouns = [x for ix, x in enumerate(nouns) if kmeans_labels[ix] == i]
        epic_verbs = [get_top_epic_verbs(n, df, k = 3) for n in cluster_nouns]
        flat_verbs = [item for sublist in epic_verbs for item in sublist]
        verb_counts = Counter(flat_verbs)
        if not verb_counts:
            results_dict[i] = coherance_score == 0.0
        else:
            coherance_score = sum([i[1] for i in verb_counts.most_common()[:n_top_words]]) / len(flat_verbs)
            results_dict[i] = coherance_score

n_clusters = 15
n_top_words = 5
kmeans_labels = KMeans(n_clusters = n_clusters).fit_predict(X)
for i in range(n_clusters):
    print("## CLUSTER {}".format(i))
    cluster_nouns = {clean_noun: original_noun for ix, (clean_noun, original_noun) in enumerate(nouns.items()) if kmeans_labels[ix] == i}
    epic_verbs = [get_top_epic_verbs(n, train_df, k = 3) for n in cluster_nouns.values()]
    flat_verbs = [item for sublist in epic_verbs for item in sublist]
    # coherance_score = len(set(flat_verbs)) / len(flat_verbs)
    verb_counts = Counter(flat_verbs)
    coherance_score = sum([i[1] for i in verb_counts.most_common()[:n_top_words]]) / len(flat_verbs)
    print("#### Cluster coherence score: {}".format(round(coherance_score, 3)))
    print("#### Top Verbs: {}".format(verb_counts.most_common()[:n_top_words]))
    for n in cluster_nouns:
        print("- {}".format(n))
    print()
    print()

# dbscan_labels = DBSCAN(eps = 10, min_samples = 2).fit(X).labels_
#
# for i in range(n_clusters):
#     print("CLUSTER {}".format(i))
#     cluster_nouns = [x for ix, x in enumerate(nouns) if dbscan_labels[ix] == i]
#     for n in cluster_nouns:
#         print("     {}".format(n))
#     print()
#     print()
