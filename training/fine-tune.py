import os
import sys
sys.path.append('..')
import pandas as pd
import regex as re


def strip_step_numbers(x):
    for ix, line in enumerate(x):
        x[ix] = re.sub(r"^\d\.?", "", line).strip()
    return x

df1 = pd.read_json("data/allrecipes.json", orient = "index")
df1["directions"] = df1["directions"].apply(lambda x: strip_step_numbers(x))
df2 = pd.read_json("data/youcookii_annotations_trainval.json", orient = "records")
records = dict(df2["database"])
all_annotations = []
for recipe in records:
    all_annotations.append([annotation["sentence"] + "." for annotation in records[recipe]["annotations"]])

directions1 = [" ".join(lines) for lines in df1["directions"].tolist()]
directions2 = [" ".join(lines) for lines in all_annotations]

with open("data/recipes.txt", "w", encoding = "utf-8") as outfile:
    for direction in directions1:
        outfile.write(direction + "\n")
    for direction in directions2:
        outfile.write(direction + "\n")

os.system(
"python3 run_language_modeling.py \
--output_dir=models/allrecipes_plus_youcookii \
--model_type=bert \
--model_name_or_path=bert-base-uncased \
--do_train \
--train_data_file=recipes.txt \
--tokenizer_name=bert-base-uncased \
--line_by_line \
--mlm"
)
