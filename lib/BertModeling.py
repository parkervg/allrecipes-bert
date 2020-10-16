import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow warnings
import sys
sys.path.append('tools/')
from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
from collections import Counter
from transformers import BertTokenizer, BertConfig, TFBertForMaskedLM
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from custom_logging import Blogger
logger = Blogger()

# TODO:
# Create simple count distribution of verbs over nouns in EpicKitchen
class BertModel():
    """
    Loads a given model for use in clustering and MLM.
    """
    def __init__(self,
                 model_dir,  # Pre-trained/fine-tuned model directory
                 model_name, # Existing HuggingFace model, e.g. 'bert-base-uncased'
                 config,
                 tokenizer
                 ):
        self.config = config if config else BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_dir = model_dir
        self.model_name = model_name
        if (self.model_dir and self.model_name):
            raise ValueError("Both model_name and model_dir can't be arguments.")
        if all(x is None for x in [self.model_dir, self.model_name]):
            raise ValueError("One of model_name or model_dir must be specified.")

    def _get_tokenizer(self):
        return self.tokenizer

    def _init_model(self):
        """
        Initializes model.
        """
        if self.model_dir:
            logger.status_update("Loading BERT model at {}...".format(self.model_dir))
            self.model = TFBertForMaskedLM.from_pretrained(self.model_dir, from_pt=True, config=self.config)
        elif self.model_name:
            logger.log("Loading BERT model {}...".format(self.model_name))
            self.model = TFBertForMaskedLM.from_pretrained(self.model_name, config=self.config)
        return self.model

class BertClustering(BertModel):
    """
    Class to cluster noun embeddings and overlay results with distribution on verbs in EpicKitchens Dataset.
    """
    def __init__(self,
                 n_clusters: int,
                 nouns: List[str],
                 verbs: List[str],
                 train_df: pd.DataFrame, # DataFrame with base_noun, base_verb columns
                 extraction_function: str = "small",
                 n_top_verbs: int = 5, # Number of verbs to list in output and take into consideration for cluster coherence score
                 model_dir: str = None,  # Pre-trained/fine-tuned model directory
                 model_name: str = None, # Existing HuggingFace model, e.g. 'bert-base-uncased'
                 config: object = None,
                 tokenizer: str = None
                 ):
        # Inherit from BertModel class
        BertModel.__init__(self, model_dir=model_dir, model_name=model_name, config=config, tokenizer=tokenizer)
        self.model = self._init_model()
        self.tokenizer = self._get_tokenizer()

        self.n_clusters = n_clusters
        self.nouns = nouns
        self.verbs = verbs
        self.train_df = train_df
        self.extraction_function = extraction_function
        if self.extraction_function == "small":
            self.extraction_function = self.extract_embedding_small
        elif self.extraction_function == "large":
            self.extraction_function = self.extract_embedding_large
        else:
            raise ValueError("Invalid extraction_function {}. Choose one of 'small', 'large'".format(self.extraction_function))
        self.n_top_verbs = n_top_verbs
        self._clean_texts()


    def _clean_texts(self):
        """
        Pre-processes nouns and verbs so that they will be more accurately encoded by BERT.
        """
        # Structure of cleaned_text:original_text
        self.nouns = {self.rearrange_colons(x): x for x in list(set(self.nouns))}
        self.verbs = {self.rearrange_colons(x) for x in list(set(self.verbs))}


    def get_noun_embeddings(self):
        """
        Creates self.noun_embeddings.
        """
        logger.status_update("Extracting noun embeddings...")
        # Creates dict of noun:noun_embedding
        self.noun_embeddings = self.convert_to_embeddings(list(self.nouns.keys()))
        return self.noun_embeddings

    def do_clustering(self):
        """
        Clusters nouns with KMeans, where self.n_clusters = k.
        Overlays each distribution with the top verbs appearing alongisde the cluster's nouns.
        """
        X = np.array(list(self.noun_embeddings.values()))
        kmeans_labels = KMeans(n_clusters=self.n_clusters).fit_predict(X)
        for i in range(self.n_clusters):
            logger.status_update("CLUSTER {}".format(i))
            cluster_nouns = {clean_noun: original_noun for ix, (clean_noun, original_noun) in enumerate(self.nouns.items()) if kmeans_labels[ix] == i}
            epic_verbs = [self.get_top_epic_verbs(n) for n in cluster_nouns.values()]
            flat_verbs = [item for sublist in epic_verbs for item in sublist]
            verb_counts = Counter(flat_verbs)
            coherance_score = sum([i[1] for i in verb_counts.most_common()[:self.n_top_verbs]]) / len(flat_verbs)
            logger.yellow("Cluster coherence score: {}".format(round(coherance_score, 3)))
            logger.yellow("Top Verbs: {}".format(verb_counts.most_common()[:self.n_top_verbs]))
            for n in cluster_nouns:
                logger.log("   {}".format(n))
            print()
            print()

    def extract_embedding_small(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Takes only last hidden layer to create embeddings with 768 dimensions
        """
        # not using tf.GradientTape will not calculate gradients
        out = self.model(tf.expand_dims(tensor, 0))
        # Only return hidden_states
        # output of all 12 layers in the encoder
        # each with shape (batch_size, sequence_length, 768)
        hidden_embed_state = out[-1]
        return tf.squeeze(tf.reduce_mean(hidden_embed_state[-1], axis=1))

    def extract_embedding_large(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Takes last 4 hidden layers to get embeddings with 3072 dimensions
        In Section 5.3, speaking on the feature-based approach:
            "The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer"
        """
        # not using tf.GradientTape will not calculate gradients
        out = self.model(tf.expand_dims(tensor, 0))
        # output of all 12 layers in the encoder
        # each with shape (batch_size, sequence_length, 768)
        hidden_states = out[-1]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = tf.concat(tuple(last_four_layers), axis=1)
        return tf.squeeze(tf.reduce_mean(cat_hidden_states, axis=1))

    def get_distance(self, text1: str, text2: str):
        """
        Gets L2 distance between vector(text1) and vector(text2).
        """
        vector1 = self.convert_to_embeddings([text1])[text1]
        vector2 = self.convert_to_embeddings([text2])[text2]
        return tf.norm(vector1 - vector2, ord="euclidean").numpy()

    def convert_to_embeddings(self, text: List[str]) -> Dict[str, tf.Tensor]:
        """
        Converts inputted text to 768 dimension embedding from BERT's hidden layers.
        """
        ids = self.tokenizer(text, add_special_tokens=True)
        embedding_dict = {t: tf.constant(i) for t, i in zip(text, ids["input_ids"])}
        embedding_dict.update((x, self.extraction_function(y)) for x, y in embedding_dict.items())
        return embedding_dict

    def get_top_epic_verbs(self, noun: str, log=False):
        """
        Gets top verbs used in association with noun from self.train_df.
        """
        existing_verbs = self.train_df.loc[self.train_df["base_noun"] == noun, "base_verb"].tolist()
        top_verbs = Counter(existing_verbs)
        if log:
            logger.yellow("Top EpicKitchen verbs for '{}':".format(noun))
            for verb, count in top_verbs.most_common()[:self.n_top_verbs]:
                logger.log("      " + verb)
            print()
        return [item[0] for item in top_verbs.most_common()[:self.n_top_verbs]]


    @staticmethod
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

    # def get_top_bert_verbs(noun: str, noun_embedding: tf.Tensor, verb_embeddings: Dict[str, tf.Tensor], k = 5):
    #     sorted_verbs = {verb: tf.norm(noun_embedding - verb_embedding, ord = "euclidean") for verb, verb_embedding in verb_embeddings.items()}
    #     sorted_verbs = sorted(sorted_verbs.items(), key = operator.itemgetter(1))
    #     logger.status_update("Top BERT verbs for '{}':".format(noun))
    #     for verb, score in sorted_verbs[:k]:
    #         logger.log("      " + verb)
    #     print()
    #     print()
    #     return [item[0] for item in sorted_verbs[:k]]


class BertMLM(BertModel):
    def __init__(self,
                 model_dir: str = None,  # Pre-trained/fine-tuned model directory
                 model_name: str = None, # Existing HuggingFace model, e.g. 'bert-base-uncased'
                 config: object = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = False),
                 tokenizer: str = None
                 ):
        # Inherit from BertModel class
        BertModel.__init__(self, model_dir=model_dir, model_name=model_name, config=config, tokenizer=tokenizer)
        self.model = self._init_model()
        self.tokenizer = self._get_tokenizer()

    def predict_mask(self, text: str, k: int = 1):
        tokenized_text = self.tokenizer.tokenize(text)
        if len([tok for tok in tokenized_text if tok=='[MASK]']) > 1:
            raise ValueError("Only one token can be masked in inputted text.")
        if tokenized_text[0] != "[CLS]":
            tokenized_text.insert(0, "[CLS]")
        if tokenized_text[-1] != "[SEP]":
            tokenized_text.append("[SEP]")
        masked_index = tokenized_text.index('[MASK]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        tokens_tensors = tf.convert_to_tensor([indexed_tokens])
        segments_tensors = tf.convert_to_tensor([segments_ids])

        pred = self.model(tokens_tensors, segments_tensors)
        if k == 1:
            predicted_index = tf.argmax(pred[0][0][masked_index])
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            return predicted_token
        else:
            predicted_indices = tf.nn.top_k(pred[0][0][masked_index], k=k, sorted=True).indices
            predicted_tokens = [self.tokenizer.convert_ids_to_tokens([predicted_index])[0] for predicted_index in predicted_indices]
            return predicted_tokens

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
