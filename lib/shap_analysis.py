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
import tensorflow as tf
from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
from sklearn.metrics import accuracy_score
from BertModeling import BertEmbeddings, BertClustering, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.pipeline import Pipeline
stop_verbs = ["put", "take"]

def get_top_epic_verbs(noun, n_top_verbs):
    existing_verbs = train_df.loc[train_df["base_noun"] == noun, "base_verb"].tolist()
    top_verbs = Counter(existing_verbs)
    return [item[0] for item in top_verbs.most_common()[:n_top_verbs]]

def verb_ix_to_text(ix):
  return Y_decoder[ix]

def noun_ix_to_text(ix):
  return X[ix]["noun"]

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

# Creating embeddings
BE = BertEmbeddings(model_dir="models/allrecipes_plus_youcookii")
noun_embeddings = BE.convert_to_embeddings(nouns)

# Training XGBoost
Y = []
good_ixs = []
for ix, i in enumerate(noun_embeddings.keys()):
  top = [i for i in get_top_epic_verbs(i, 10) if i not in stop_verbs]
  if top:
      Y.append(top[0])
      good_ixs.append(ix)
Y_encoder = {x:ix for ix, x in enumerate(list(set(Y)))}
Y_decoder = {ix:x for ix, x in enumerate(list(set(Y)))}
X = {ix: {"noun": k, "embedding": v} for ix, (k, v) in enumerate(noun_embeddings.items()) if ix in good_ixs}
Xtrain, Xtest, Ytrain, Ytest = train_test_split(list(X.keys()), [Y_encoder[y] for y in Y], test_size=0.33, random_state=23)
Xtrain_embeds = np.array([X[i]["embedding"] for i in Xtrain])
Xtest_embeds = np.array([X[i]["embedding"] for i in Xtest])
model = XGBClassifier()
model.fit(Xtrain_embeds, Ytrain)
y_pred = model.predict(Xtest_embeds)

accuracy = accuracy_score(Ytest, y_pred)
print("Accuracy: {}".format(accuracy))


for ix, (pred, act) in enumerate(zip([Y_decoder[i] for i in y_pred], [Y_decoder[i] for i in Ytest])):
  print(Xtest[ix])
  print("Actual: {}".format(act))
  print("Predicted: {}".format(pred))
  print()
  print()


def subspace_score(word: tf.Tensor, dims: List[int]) -> float:
  """
  As defined in Jang et al.
  """
  slice = tf.gather(word, indices=dims)
  sum_slice = tf.reduce_sum(slice, axis=0)
  return (sum_slice / len(dims)).numpy()

def find_top_dimensions(shap_array, predictors, predicted, k = 20):
  # Dimension SHAP values across all y
  dim_shap_values = defaultdict(list)
  # Dimension SHAP values, divided across y
  # Answers question:
  #   For each noun predicting a verb's probability from the noun's embedding,
  #   which dimensions were the most influential
  verb_shap_values = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # noun_ix: verb_ix: dim_ix: avg([values])
  for verb_ix in range(shap_array.shape[0]):
    for noun_ix in range(shap_array.shape[1]):
      for dim_ix in range(shap_array.shape[2]):
        val = shap_array[verb_ix][noun_ix][dim_ix]
        if val != 0.0:
          verb_shap_values[noun_ix][verb_ix][dim_ix] = val
          # dim_shap_values[dim_ix].append(val)
          # verb_shap_values[verb_ix][dim_ix].append(val)

  # Looking through verb_shap_values for those instances aligning with y_pred
  top_dimensions_all = defaultdict(list) # y, sorted(dim_ix)
  top_dimensions_no_repeats = defaultdict(set) # y, sorted(dim_ix)
  for x, y in zip(predictors, predicted):
    all_dims = verb_shap_values[x][y]
    if all_dims:
      sorted_dims = [i[0] for i in sorted(all_dims.items(), key = lambda x: x[1], reverse = True) if i[1] > 0.0][:k]
      for dim in sorted_dims:
        top_dimensions_all[y].append(dim)
        top_dimensions_no_repeats[y].add(dim)

  dim_distribution_all = [v for v in top_dimensions_all.values()]
  dim_distribution_all = [item for sublist in dim_distribution_all for item in sublist]
  dim_distribution_all = Counter(dim_distribution_all)

  dim_distribution_no_repeats = [list(v) for v in top_dimensions_no_repeats.values()]
  dim_distribution_no_repeats = [item for sublist in dim_distribution_no_repeats for item in sublist]
  dim_distribution_no_repeats = Counter(dim_distribution_no_repeats)

  return {"all": dim_distribution_all, "no_repeats": dim_distribution_no_repeats, "raw": top_dimensions_all}


shap_values = shap.TreeExplainer(model).shap_values(list(noun_embeddings.values()))
shap_array = np.array(shap_values)
dim_distribution = find_top_dimensions(shap_array, list(X.keys()), model.classes_, k = 30)

for k, v in dim_distribution["raw"].items():
  counts = Counter(v)
  top = [x for x in counts.most_common()[:10] if x[1] > 1]
  print("For label '{}': ".format(verb_ix_to_text(k)))
  print("Top discriminatory dimensions:")
  for t in top:
    print("   {}".format(t))
  print("Words with highest subspace scores: ")
  scores = [(noun, subspace_score(embed, v)) for noun, embed in noun_embeddings.items()]
  for (word, score) in sorted(scores, key = lambda x: x[1], reverse = True)[:5]:
    print("    {}".format((word,score)))
  print()
  print()



  dim_distribution["all"].most_common()[:20]
