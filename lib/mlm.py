import sys
sys.path.append('tools/')
from BertModeling import BertMLM
from custom_logging import Blogger
logger = Blogger()


def masked_language_predict(text, k=None):
    BMLM = BertMLM(model_dir="models/allrecipes/checkpoint-3500")
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
    text = sys.argv[1]
    if len(sys.argv) > 2:
        k = int(sys.argv[2])
        masked_language_predict(text, k)
    else:
        masked_language_predict(text)
