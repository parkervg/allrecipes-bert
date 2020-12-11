import os
import sys
import argparse
from lib.BertModeling import BertMLM
from tools.custom_logging import Blogger

logger = Blogger()


def masked_language_predict_prob(text, word):
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
            text = input("Input text: ")
            word = input("Input word: ")
        logger.status_update(BMLM.predict_mask_prob(text, word))
        text = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bert masked language modelling probabilities"
    )
    parser.add_argument("text", help="the input text to predict on")
    parser.add_argument("word", help="the word to fill the mask")
    args = parser.parse_args()
    masked_language_predict_prob(args.text, args.word)
