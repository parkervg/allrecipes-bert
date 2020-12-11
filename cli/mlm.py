import os
import sys
import argparse
from lib.BertModeling import BertMLM
from tools.custom_logging import Blogger

logger = Blogger()


def masked_language_predict(text, k=None):
    if os.path.exists("models/allrecipes_plus_youcookii"):
        BMLM = BertMLM(model_dir="models/allrecipes_plus_youcookii")
    else:
        logger.red(
            "Can't find fine-tuned model at models/allrecipes_plus_youcookii. Using generic bert-base-uncased from HuggingFace instead."
        )
        BMLM = BertMLM(model_name="bert-base-uncased")
    while True:
        if not text:
            print()
            text = input("Input: ")
        if k:
            logger.status_update(BMLM.predict_mask(text, k))
        else:
            logger.status_update(BMLM.predict_mask(text))
        text = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bert masked language modelling.")
    parser.add_argument("text", help="the input text to predict on")
    parser.add_argument("-k", type=int, default=1, help="number of results to return")
    args = parser.parse_args()
    masked_language_predict(args.text, args.k)
