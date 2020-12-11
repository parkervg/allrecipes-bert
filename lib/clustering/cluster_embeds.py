import sys

sys.path.append("tools/")
sys.path.append("lib/")
import os
import pandas as pd
from BertModeling import BertClustering
from pre_process_data import pre_process_data
from custom_logging import Blogger

logger = Blogger()


def cluster_embeds(n_clusters, noun_cap=None):
    verbs, nouns, train_df = pre_process_data()

    if os.path.exists("models/allrecipes_plus_youcookii"):
        BC = BertClustering(
            n_clusters,
            nouns,
            verbs,
            train_df,
            model_dir="models/allrecipes_plus_youcookii",
        )
    else:
        logger.red(
            "Can't find fine-tuned model at models/allrecipes_plus_youcookii. Using generic bert-base-uncased from HuggingFace instead."
        )
        BC = BertClustering(
            n_clusters, nouns, verbs, train_df, model_name="bert-base-uncased"
        )
    BC.do_clustering()


if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    noun_cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cluster_embeds(n_clusters, noun_cap)
