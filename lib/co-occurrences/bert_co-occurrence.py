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

from collections import defaultdict
from BertModeling import BertMLM
from custom_logging import Blogger
from pre_process_data import pre_process_data
from custom_logging import Blogger
logger = Blogger()

def verb_mask(noun: str):
    return "now let me show you how to [MASK] the {}.".format(noun)

def noun_mask(verb: str):
    return "now let me show you how to {} the [MASK].".format(verb)

def rearrange_colons(text):
    """
    Handles instances like 'mitt:oven' in EpicKitchen to create 'oven mitt'.
    """
    text = text.split(":")
    if len(text) > 1:
        text = text[1] + " " + text[0]
    else:
        text = text[0]
    return text

def calculate_bert_cooccurrences():
    """
    This took like 3 hours to run in Google Colab. Beware.
    """
    verbs, nouns, _ = pre_process_data()

    BMLM = BertMLM(model_dir="models/allrecipes_plus_youcookii")

    prob_dict = defaultdict(lambda: defaultdict(int))
    for verb in verbs:
        for noun in nouns:
            v_prob = BMLM.predict_mask_prob(verb_mask(noun), verb)
            n_prob = BMLM.predict_mask_prob(noun_mask(verb), noun)
            # Assign prob as mean between two masked language instances
            prob_dict[verb][noun] = statistics.mean([v_prob, n_prob])

    cooc_df = pd.DataFrame(index = verbs, columns=nouns)
    for verb in coocc_dict:
        for noun, noun_count in coocc_dict[verb].items():
            cooc_df[noun][cooc_df.index == verb] = noun_count
    cooc_df = cooc_df.fillna(0)

df = pd.read_csv("data/bert_co-occurrence.csv", index_col=0)
max_softmax = max(df.max().tolist())

# start_index = 1
# df_chunk_1 = df[start_index:start_index+50][df.columns[start_index:start_index+75]]
# cmap = sns.light_palette((260, 75, 60), input="husl", n_colors = 20)
# dimensions = (40, 25)
# fig, ax = plt.subplots(figsize=dimensions)
# ax = sns.heatmap(data = df_chunk_1,
#                  vmin = 0,
#                  vmax = max_softmax+0.1,
#                  cmap = cmap,
#                  linecolor='black',
#                  linewidths=.01)
# ax.text(30, -8, "Average Softmax across Verb/Noun Co-Occurence", fontsize = 20, weight='bold')
# ax.text(34, -6, "'now let me show you how to [MASK] the [MASK].'", fontsize = 13, style='italic')# Set ticks to all sides
# ax.tick_params(right=True, top=True, labelright=True, labeltop=True, rotation=0, labelsize=14)
# #Rotate X ticks
# plt.xticks(rotation='vertical')
# #plt.show()
# plt.savefig("visualizations/bert_co-occurence.png", dpi=500, bbox_inches="tight", pad_inches=.5)


start_index = 1
df_chunk_1 = df[start_index:start_index+25][df.columns[start_index:start_index+50]]
cmap = sns.light_palette((260, 75, 60), input="husl", n_colors = 30)
dimensions = (40, 25)
fig, ax = plt.subplots(figsize=dimensions)
ax = sns.heatmap(data = df_chunk_1,
                 vmin = 0,
                 vmax = max_softmax,
                 cmap = cmap,
                 linecolor='black',
                 linewidths=.01)
ax.text(18, -5, "Average Softmax across Verb/Noun Co-Occurence", fontsize = 24, weight='bold')
ax.text(21, -4, "'now let me show you how to [MASK] the [MASK].'", fontsize = 17, style='italic')# Set ticks to all sides
ax.tick_params(right=True, top=True, labelright=True, labeltop=True, rotation=0, labelsize=16)
#Rotate X ticks
plt.xticks(rotation='vertical')
#plt.show()
plt.savefig("visualizations/bert_co-occurence_small1.png", dpi=500, bbox_inches="tight", pad_inches=.5)


cmap = sns.light_palette((260, 75, 60), input="husl", n_colors = 20)
dimensions = (40, 25)
fig, ax = plt.subplots(figsize=dimensions)
ax = sns.heatmap(data = df,
                 vmin = 0,
                 vmax = max_softmax+0.1,
                 cmap = cmap,
                 linecolor='black',
                 linewidths=.01)
ax.text(30, -8, "Average Softmax across Verb/Noun Co-Occurence", fontsize = 20, weight='bold')
ax.text(34, -6, "'now let me show you how to [MASK] the [MASK].'", fontsize = 13, style='italic')# Set ticks to all sides
ax.tick_params(right=True, top=True, labelright=True, labeltop=True, rotation=0, labelsize=14)
#Rotate X ticks
plt.xticks(rotation='vertical')
plt.show()
