from compare_embeddings import convert_to_token_tensors, extract_embedding_small
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel, BertConfig, BertModel
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import mpld3

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
model = TFBertModel.from_pretrained('bert-base-uncased', config = config)

df = pd.read_csv("EPIC_train_action_labels.csv")
n_verbs = 50
n_nouns = 50
verbs = list(set([x.split(":")[-1].replace("-", " ") for x in df.verb.tolist()]))[100: 100 + n_verbs]
nouns = list(set([x.split(":")[-1].replace("-", " ") for x in df.noun.tolist()]))[100: 100 + n_nouns]

verb_tensors = convert_to_token_tensors(verbs)
noun_tensors = convert_to_token_tensors(nouns)
duplicates = [x for x in noun_tensors if x in verb_tensors]
for dup in duplicates:
    noun_tensors.pop(dup)
    verb_tensors.pop(dup)
n_verbs, n_nouns = len(noun_tensors), len(verb_tensors)
d = verb_tensors.copy()
d.update(noun_tensors)

text = list(d.keys())
tensors = list(d.values())
X = np.array([extract_embedding_small(tensor, model) for tensor in tensors])

pca = PCA(n_components=2)
pca_X = pca.fit_transform(X)
x = pca_X[:, 0]
y = pca_X[:, 1]
len(pca_X)

fig, ax = plt.subplots(figsize = (14,8))
for i in range(n_verbs):
    ax.plot(pca_X[i, 0], pca_X[i, 1], "or", ms = 10, alpha = 0.2, c = "blue")
for i in range(n_verbs, n_verbs + n_nouns):
    ax.plot(pca_X[i, 0], pca_X[i, 1], "or", ms = 10, alpha = 0.2, c = "red")
#ax.plot(pca_X[:, 0], pca_X[:, 1], "or", ms = 10, alpha = 0.2, c = "blue")
ax.set_xlim(-2, 0)
for i, txt in enumerate(text):
    plt.annotate(txt, (pca_X[:, 0][i], pca_X[:, 1][i]))
mpld3.display(fig)
