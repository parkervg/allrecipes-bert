import sys
sys.path.append('tools/')
import os
import pandas as pd
from BertModeling import BertClustering
from custom_logging import Blogger
logger = Blogger()

def cluster_embeds(n_clusters, noun_cap=None):
    # Loading in verbs and nouns
    train_df = pd.read_csv("data/EPIC_train_action_labels.csv")
    noun_classes_df = pd.read_csv("data/EPIC_noun_classes.csv")
    nouns = noun_classes_df.class_key.tolist()[:noun_cap] if noun_cap else noun_classes_df.class_key.tolist()
    verb_classes_df = pd.read_csv("data/EPIC_verb_classes.csv")
    verbs = verb_classes_df.class_key.tolist()

    # Joining on ids from noun/verb class dfs
    # NOTE: some base classes seem a bit odd, like "door" --> "cupboard"?
    #   You can open/close both, but "put" (in) cupboard, not with door
    train_df["base_noun"] = train_df.join(noun_classes_df.set_index("noun_id"), on="noun_class")["class_key"].tolist()
    train_df["base_verb"] = train_df.join(verb_classes_df.set_index("verb_id"), on="verb_class")["class_key"].tolist()
    if os.path.exists("models/allrecipes_plus_youcookii"):
        BC = BertClustering(n_clusters, nouns, verbs, train_df, model_dir="models/allrecipes_plus_youcookii")
    else:
        logger.red("Can't find fine-tuned model at models/allrecipes_plus_youcookii. Using generic bert-base-uncased from HuggingFace instead.")
        BC = BertClustering(n_clusters, nouns, verbs, train_df, model_name="bert-base-uncased")
    BC.do_clustering()


if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    noun_cap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cluster_embeds(n_clusters, noun_cap)
