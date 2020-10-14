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
from custom_logging import Blogger
logger = Blogger()

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
model = TFBertForMaskedLM.from_pretrained("models/allrecipes/checkpoint-3500", from_pt = True, config = config)

df = pd.read_csv("example.csv")
# verbs = list(set([x.split(":")[-1].replace("-", " ") for x in df.verb.tolist()]))
nouns = list(set([x.split(":")[-1].replace("-", " ") for x in df.noun.tolist()]))

# Transforming to embeddings
# verb_embeddings = convert_to_embeddings(verbs, model = model)
noun_embeddings = convert_to_embeddings(nouns, model = model)
X = np.array(list(noun_embeddings.values()))


n_clusters = 10
n_top_words = 5
kmeans_labels = KMeans(n_clusters = n_clusters).fit_predict(X)

for i in range(n_clusters):
    print("CLUSTER {}".format(i))
    cluster_nouns = [x for ix, x in enumerate(nouns) if kmeans_labels[ix] == i]
    epic_verbs = [get_top_epic_verbs(n, df, k = 3) for n in cluster_nouns]
    flat_verbs = [item for sublist in epic_verbs for item in sublist]
    # coherance_score = len(set(flat_verbs)) / len(flat_verbs)
    verb_counts = Counter(flat_verbs)
    coherance_score = sum([i[1] for i in verb_counts.most_common()[:n_top_words]]) / len(flat_verbs)
    print("Cluster coherence score: {}".format(round(coherance_score, 3)))
    print("Top Verbs: {}".format(verb_counts.most_common()[:n_top_words]))
    print("___________________________________________________")
    for n in cluster_nouns:
        print("     {}".format(n))
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
