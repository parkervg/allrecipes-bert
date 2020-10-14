from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
from transformers import BertTokenizer, TFBertModel, BertConfig, BertModel, TFBertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import tensorflow as tf
import pandas as pd
import sys
from collections import Counter
import operator
from custom_logging import Blogger
logger = Blogger()


# config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
# model = TFBertModel.from_pretrained('bert-base-uncased', config = config)
# from transformers import TFBertForMaskedLM
# config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
# tuned_model = TFBertForMaskedLM.from_pretrained("models/allrecipes/checkpoint-3500", from_pt = True, config = config)
# get_distance("cook", "steak", tuned_model, extract_embedding_small)
# get_distance("dog", "cat", tuned_model, extract_embedding_small)
#
# get_distance("cook", "steak", model, extract_embedding_small)
# get_distance("heat", "oven", model, extract_embedding_small)
#
# toks = convert_to_token_tensors(["tomato"])["tomato"]
# out = tuned_model(tf.expand_dims(toks, 0))
# out[0].shape
# len(out[1])
# text = ["tomato"]
# ids = tokenizer(text, add_special_tokens = True)
# embedding_dict = {t: tf.constant(i) for t, i in zip(text, ids["input_ids"])}
# tensor = embedding_dict["tomato"]


def extract_embedding_small(tensor: tf.Tensor, model: object) -> tf.Tensor:
    """
    Takes only last hidden layer to create embeddings with 768 dimensions
    """
    # not using tf.GradientTape will not calculate gradients
    out = model(tf.expand_dims(tensor, 0))
    # Only return hidden_states
    # output of all 12 layers in the encoder
    # each with shape (batch_size, sequence_length, 768)
    hidden_embed_state = out[-1]
    return tf.squeeze(tf.reduce_mean(hidden_embed_state[-1], axis = 1))

def extract_embedding_large(tensor: tf.Tensor, model: object) -> tf.Tensor:
    """
    Takes last 4 hidden layers to get embeddings with 3072 dimensions
    In Section 5.3, speaking on the feature-based approach:
        "The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer"
    """
    # not using tf.GradientTape will not calculate gradients
    out = model(tf.expand_dims(tensor, 0))
    # output of all 12 layers in the encoder
    # each with shape (batch_size, sequence_length, 768)
    hidden_states = out[-1]
    last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
    cat_hidden_states = tf.concat(tuple(last_four_layers), axis = 1)
    return tf.squeeze(tf.reduce_mean(cat_hidden_states, axis = 1))

def get_distance(text1: str, text2: str, model: object, extraction_func = extract_embedding_large):
    vector1 = extraction_func(convert_to_token_tensors([text1])[text1], model = model)
    vector2 = extraction_func(convert_to_token_tensors([text2])[text2], model = model)
    return tf.norm(vector1 - vector2, ord = "euclidean").numpy()


def get_top_bert_verbs(noun: str, noun_embedding: tf.Tensor, verb_embeddings: Dict[str, tf.Tensor], k = 5):
    sorted_verbs = {verb: tf.norm(noun_embedding - verb_embedding, ord = "euclidean") for verb, verb_embedding in verb_embeddings.items()}
    sorted_verbs = sorted(sorted_verbs.items(), key = operator.itemgetter(1))
    logger.status_update("Top BERT verbs for '{}':".format(noun))
    for verb, score in sorted_verbs[:k]:
        logger.log("      " + verb)
    print()
    print()
    return [item[0] for item in sorted_verbs[:k]]

def get_top_epic_verbs(noun: str, df: pd.DataFrame, k = 5, log = False):
    existing_verbs = df.loc[df["base_noun"] == noun, "base_verb"].tolist()
    top_verbs = Counter(existing_verbs)
    if log:
        logger.yellow("Top EpicKitchen verbs for '{}':".format(noun))
        for verb, count in top_verbs.most_common()[:k]:
            logger.log("      " + verb)
        print()
    return [item[0] for item in top_verbs.most_common()[:k]]

# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
def convert_to_embeddings(text: List[str], model: object, extraction_func = extract_embedding_small) -> Dict[str, tf.Tensor]:
    ids = tokenizer(text, add_special_tokens = True)
    embedding_dict = {t: tf.constant(i) for t, i in zip(text, ids["input_ids"])}
    embedding_dict.update((x, extraction_func(y, model = model)) for x, y in embedding_dict.items())
    return embedding_dict


if __name__ == "__main__":
    if len(sys.argv) == 3:
        csv_location = sys.argv[1]
        k = int(sys.argv[2])
        noun_cap = None
    else:
        csv_location = sys.argv[1]
        k = int(sys.argv[2])
        noun_cap = int(sys.argv[3])

    logger.log(csv_location)
    logger.log(k)
    logger.log(noun_cap)
    logger.log("Loading BERT model...")
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model = TFBertForMaskedLM.from_pretrained("models/allrecipes/checkpoint-3500", from_pt = True, config = config)

    # Parsing input data
    df = pd.read_csv(csv_location)
    verbs = list(set([x.split(":")[-1].replace("-", " ") for x in df.verb.tolist()]))

    if noun_cap:
        nouns = list(set([x.split(":")[-1].replace("-", " ") for x in df.noun.tolist()]))[50:50 + noun_cap]
    else:
        nouns = list(set([x.split(":")[-1].replace("-", " ") for x in df.noun.tolist()]))

    # Transforming to embeddings
    verb_embeddings = convert_to_embeddings(verbs, model = model)
    noun_embeddings = convert_to_embeddings(nouns, model = model)

    for noun in nouns:
        bert_verbs = get_top_bert_verbs(noun, noun_embeddings[noun], verb_embeddings, k = k)
        epic_verbs = get_top_epic_verbs(noun, df, k = k)
    #    alignment_report(bert_verbs, epic_verbs)




# Issues:
# Polysemy of a word like 'foil'
# Fine-tuning should help
# https://docs.google.com/document/d/1i_4Uys55-ZDUNM9aqU3waGy_uibEgLlqSjADw1ABYaE/edit?ts=5f7cb2ff
# Possible use case:
#      - Identify verb and noun in sentence, use masked language model to predict
#      - Use similarity to gauge whether it fits into our existing epickitchen data

"""
Steps as state sequences
    - Ingredients/Ingredients w/ actions
        - What's the life span of the currants through the recipe?
    step 1:
        place, boil, strain, discard
    step 2:
        no currants now, just juice
    For each object, index activity with temporal index
    step: Acitiviy - ti
    mix2: overlap with water and currants
        mix water and currants
    Introducing new juice object: activity --> activity result
        Have result of activities be the object itself
    Batter --> result of mixing ingredients
        - Still be able to reconstruct idea that there is sugar in Batter
        - Object inheritance
    Dependency parse to gather original entities
"""
