import sys
sys.path.append('tools/')
sys.path.append('lib/')
import os
from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import defaultdict, Counter
from BertModeling import BertMLM
from custom_logging import Blogger
from pre_process_data import pre_process_data
from sklearn.cluster import KMeans
from pre_process_data import pre_process_data
from custom_logging import Blogger
logger = Blogger()


df = pd.read_csv("data/bert_co-occurrence.csv", index_col=0)

verbs,nouns, train_df = pre_process_data()
def get_top_epic_verbs(noun: str, n_top_verbs: int, log=False):
    """
    Gets top verbs used in association with noun from self.train_df.
    """
    existing_verbs = train_df.loc[train_df["base_noun"] == noun, "base_verb"].tolist()
    top_verbs = Counter(existing_verbs)
    if log:
        logger.yellow("Top EpicKitchen verbs for '{}':".format(noun))
        for verb, count in top_verbs.most_common()[:n_top_verbs]:
            logger.log("      " + verb)
        print()
    return [item[0] for item in top_verbs.most_common()[:n_top_verbs]]


n_clusters = 15
n_top_verbs = 5
# Creating array, where one row is a noun's probability for that verb
X = np.array([df[i].tolist() for i in df.columns[1:]])
nouns = df.columns.tolist()[1:]
kmeans_labels = KMeans(n_clusters=n_clusters).fit_predict(X)
for i in range(n_clusters):
    print("## CLUSTER {}".format(i))
    cluster_nouns = {noun for ix, noun in enumerate(nouns) if kmeans_labels[ix] == i}
    epic_verbs = [get_top_epic_verbs(n, n_top_verbs) for n in cluster_nouns]
    flat_verbs = [item for sublist in epic_verbs for item in sublist]
    verb_counts = Counter(flat_verbs)
    try:
        coherence_score = sum([i[1] for i in verb_counts.most_common()[:n_top_verbs]]) / len(flat_verbs)
    except ZeroDivisionError:
        coherence_score = 0.0
    print("#### Cluster coherence score: {}".format(round(coherence_score, 3)))
    print("#### Top Verbs: {}".format(verb_counts.most_common()[:n_top_verbs]))
    for n in cluster_nouns:
        print("- {}".format(n))
    print()
    print()
