import sys
sys.path.append('lib/')
import pandas as pd
from collections import defaultdict
from pre_process_data import pre_process_data
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

"""
Creating csv with index verbs and columns nouns. Values represent the noun/verb co-occurence count.
"""

def calculate_epickitchen_cooccurrences():
    verbs, nouns, train_df = pre_process_data()
    coocc_dict = defaultdict(lambda: defaultdict(int))
    cooc_df = pd.DataFrame(index = verbs, columns=nouns)
    for noun, verb in zip(train_df["base_noun"], train_df["base_verb"]):
        coocc_dict[verb][noun] += 1
    for verb in coocc_dict:
        for noun, noun_count in coocc_dict[verb].items():
            cooc_df[noun][cooc_df.index == verb] = noun_count
    cooc_df = cooc_df.fillna(0)
    cooc_df.to_csv("data/epickitchen_co-occurrence.csv")

df = pd.read_csv("data/epickitchen_co-occurrence.csv", index_col=0)
max_count = max(df.max().tolist())

# Heatmap time
df_chunk_1 = df[:50][df.columns[:75]]
cmap = sns.light_palette((260, 75, 60), input="husl", n_colors = 20)
dimensions = (40, 25)
fig, ax = plt.subplots(figsize=dimensions)
ax = sns.heatmap(data = df_chunk_1,
                 vmin = 0,
                 vmax = max_count+1,
                 cmap = cmap,
                 linecolor='black',
                 linewidths=.01)
ax.text(30, -6, "Verb/Noun Co-Occurrences in EpicKitchem", fontsize = 20, weight='bold')
#ax.text(37, -6, "'now let me show you how to [MASK] the [MASK].'", fontsize = 13, style='italic')
# Set ticks to all sides
ax.tick_params(right=True, top=True, labelright=True, labeltop=True, rotation=0, labelsize=14)
#Rotate X ticks
plt.xticks(rotation='vertical')
#plt.show()
plt.savefig("visualizations/epickitchen_co-occurence.png", dpi=500, bbox_inches="tight", pad_inches=.5)
