import sys
sys.path.append('tools/')
sys.path.append('lib/')
import pandas as pd
from collections import Counter, defaultdict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
import shap
from sklearn.metrics import accuracy_score
from BertModeling import BertEmbeddings, BertClustering, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.pipeline import Pipeline
stop_verbs = ["put", "take"]


import tensorflow as tf
BM = BertModel(model_dir = "models/allrecipes_plus_youcookii")
model = BM._init_model()
tokenizer = BM._get_tokenizer()

word = "dice"
text = "now let me show you how to [MASK] the tomato"
tokenized_text = tokenizer.tokenize(text)
if len([tok for tok in tokenized_text if tok=='[MASK]']) > 1:
    raise ValueError("Only one token can be masked in inputted text.")
if tokenized_text[0] != "[CLS]":
    tokenized_text.insert(0, "[CLS]")
if tokenized_text[-1] != "[SEP]":
    tokenized_text.append("[SEP]")
masked_index = tokenized_text.index('[MASK]')

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0] * len(tokenized_text)

tokens_tensors = tf.convert_to_tensor([indexed_tokens])
segments_tensors = tf.convert_to_tensor([segments_ids])

pred = model(tokens_tensors, segments_tensors)

word = "cut"
tokenized_target_word = tokenizer.tokenize(word)
if len(tokenized_target_word) > 1:
    raise ValueError("The target word is comprised of more than one tokens: {}".format(tokenized_target_word))
target_word_index = tokenizer.convert_tokens_to_ids(tokenized_target_word)[0]
softmax_probs = tf.nn.softmax(pred[0][0][masked_index], axis=0) # So that the number can be interpreted as a probability
return softmax_probs[target_word_index].numpy()

pred[0][0][masked_index][target_word_index]


BMLM = BertMLM(model_dir="models/allrecipes_plus_youcookii")
BMLM.predict_mask_prob(text, word)






# noun_cap = None
# # Loading in verbs and nouns
# train_df = pd.read_csv("data/EPIC_train_action_labels.csv")
# noun_classes_df = pd.read_csv("../epic-kitchens-55-annotations/EPIC_noun_classes.csv")
# nouns = noun_classes_df.class_key.tolist()[:noun_cap] if noun_cap else noun_classes_df.class_key.tolist()
# verb_classes_df = pd.read_csv("../epic-kitchens-55-annotations/EPIC_verb_classes.csv")
# verbs = verb_classes_df.class_key.tolist()
#
# # Joining on ids from noun/verb class dfs
# # NOTE: some base classes seem a bit odd, like "door" --> "cupboard"?
# #   You can open/close both, but "put" (in) cupboard, not with door
# train_df["base_noun"] = train_df.join(noun_classes_df.set_index("noun_id"), on="noun_class")["class_key"].tolist()
# train_df["base_verb"] = train_df.join(verb_classes_df.set_index("verb_id"), on="verb_class")["class_key"].tolist()
# n_clusters = 20
#
#
# BE = BertEmbeddings(model_dir="models/allrecipes/checkpoint-3500")
# tv = BE.convert_to_embeddings(["tomato"])["tomato"]
# tv.shape
# import tensorflow as tf
# tv = tf.gather(tv, indices = test)
# tv.shape
# tf.reduce_sum(tv, axis=0)
#


# for n in ["rack:drying", "risotto", "bacon", "trouser", "heart", "leek", "cloth"]:
#     print(get_top_epic_verbs(n, 3))
#
# def get_top_epic_verbs(noun, n_top_verbs):
#     existing_verbs = train_df.loc[train_df["base_noun"] == noun, "base_verb"].tolist()
#     top_verbs = Counter(existing_verbs)
#     return [item[0] for item in top_verbs.most_common()[:n_top_verbs]]

# def verb_ix_to_text(ix):
#   return Y_decoder[ix]
#
# def noun_ix_to_text(ix):
#   return X[ix]["noun"]
#
#
#
#
#
# def find_top_dimensions(shap_array, predictors, predicted, k = 20):
#   # Dimension SHAP values across all y
#   dim_shap_values = defaultdict(list)
#   # Dimension SHAP values, divided across y
#   # Answers question:
#   #   For each noun predicting a verb's probability from the noun's embedding,
#   #   which dimensions were the most influential
#   verb_shap_values = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # noun_ix: verb_ix: dim_ix: avg([values])
#   for verb_ix in range(shap_array.shape[0]):
#     for noun_ix in range(shap_array.shape[1]):
#       for dim_ix in range(shap_array.shape[2]):
#         val = shap_array[verb_ix][noun_ix][dim_ix]
#         if val != 0.0:
#           verb_shap_values[noun_ix][verb_ix][dim_ix] = val
#           # dim_shap_values[dim_ix].append(val)
#           # verb_shap_values[verb_ix][dim_ix].append(val)
#
#   # Looking through verb_shap_values for those instances aligning with y_pred
#   top_dimensions_all = defaultdict(list) # y, sorted(dim_ix)
#   top_dimensions_no_repeats = defaultdict(set) # y, sorted(dim_ix)
#   for x, y in zip(predictors, predicted):
#     all_dims = verb_shap_values[x][y]
#     if all_dims:
#       sorted_dims = [i[0] for i in sorted(all_dims.items(), key = lambda x: x[1], reverse = True) if i[1] > 0.0][:k]
#       for dim in sorted_dims:
#         top_dimensions_all[y].append(dim)
#         top_dimensions_no_repeats[y].add(dim)
#
#   dim_distribution_all = [v for v in top_dimensions_all.values()]
#   dim_distribution_all = [item for sublist in dim_distribution_all for item in sublist]
#   dim_distribution_all = Counter(dim_distribution_all)
#
#   dim_distribution_no_repeats = [list(v) for v in top_dimensions_no_repeats.values()]
#   dim_distribution_no_repeats = [item for sublist in dim_distribution_no_repeats for item in sublist]
#   dim_distribution_no_repeats = Counter(dim_distribution_no_repeats)
#
#   return {"all": dim_distribution_all, "no_repeats": dim_distribution_no_repeats}
#
