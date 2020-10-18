import sys
sys.path.append('tools/')
import os
from BertModeling import BertMLM
from custom_logging import Blogger
logger = Blogger()


def masked_language_predict(text, k=None):
    if os.path.exists("models/allrecipes_plus_youcookii"):
        BMLM = BertMLM(model_dir="models/allrecipes_plus_youcookii")
    else:
        logger.red("Can't find fine-tuned model at models/allrecipes_plus_youcookii. Using generic bert-base-uncased from HuggingFace instead.")
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
    if len(sys.argv) > 1:
        text = sys.argv[1]
        k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        text = None
        k = None
    masked_language_predict(text, k)
