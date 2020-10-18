import pandas as pd

def pre_process_data():
    # Pre-processing text
    train_df = pd.read_csv("data/EPIC_train_action_labels.csv")
    noun_classes_df = pd.read_csv("data/EPIC_noun_classes.csv")
    nouns = noun_classes_df.class_key.tolist()
    verb_classes_df = pd.read_csv("data/EPIC_verb_classes.csv")
    verbs = verb_classes_df.class_key.tolist()

    # Joining on ids from noun/verb class dfs
    # NOTE: some base classes seem a bit odd, like "door" --> "cupboard"?
    #   You can open/close both, but "put" (in) cupboard, not with door
    train_df["base_noun"] = train_df.join(noun_classes_df.set_index("noun_id"), on="noun_class")["class_key"].tolist()
    train_df["base_verb"] = train_df.join(verb_classes_df.set_index("verb_id"), on="verb_class")["class_key"].tolist()

    return verbs, nouns, train_df
