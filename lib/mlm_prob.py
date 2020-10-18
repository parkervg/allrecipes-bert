import sys
sys.path.append('tools/')
import os
from BertModeling import BertMLM
from custom_logging import Blogger
logger = Blogger()


def masked_language_predict_prob(text, word):
    if os.path.exists("models/allrecipes_plus_youcookii"):
        BMLM = BertMLM(model_dir="models/allrecipes_plus_youcookii")
    else:
        logger.red("Can't find fine-tuned model at models/allrecipes_plus_youcookii. Using generic bert-base-uncased from HuggingFace instead.")
        BMLM = BertMLM(model_name="bert-base-uncased")
    while True:
        if not text:
            print()
            text = input("Input: ")
        logger.status_update(BMLM.predict_mask_prob(text, word))
        text = None

if __name__ == "__main__":
    text = sys.argv[1]
    word = sys.argv[2]
    masked_language_predict_prob(text, word)
