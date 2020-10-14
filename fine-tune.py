import os
import pandas as pd
import regex as re


def strip_step_numbers(x):
    for ix, line in enumerate(x):
        x[ix] = re.sub(r"^\d\.?", "", line).strip()
    return x

df = pd.read_json("allrecipes.json", orient = "index")
df["directions"] = df["directions"].apply(lambda x: strip_step_numbers(x))
directions = [" ".join(lines) for lines in df["directions"].tolist()][:10000]
with open("recipes.txt", "w", encoding = "utf-8") as outfile:
    for direction in directions:
        outfile.write(direction + "\n")

os.system(
"python3 run_language_modeling.py \
--output_dir=models/allrecipes \
--model_type=bert \
--model_name_or_path=bert-base-uncased \
--do_train \
--train_data_file=recipes.txt \
--line_by_line \
--mlm"
)
