from typing import (Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable)
from transformers import BertTokenizer, TFBertModel, BertConfig, BertModel, TFBertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import tensorflow as tf

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = False)
model = TFBertForMaskedLM.from_pretrained("models/allrecipes/checkpoint-3500", from_pt = True, config = config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


text = """ [CLS] Place the salmon fillet, skin side down, on the baking sheet; [MASK] the knife and cut the salmon into small pieces  [SEP]"""

masked_language_predict(text, k =3)

def masked_language_predict(text, k = 1):
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index('[MASK]')

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    tokens_tensors = tf.convert_to_tensor([indexed_tokens])
    segments_tensors = tf.convert_to_tensor([segments_ids])

    pred = model(tokens_tensors, segments_tensors)
    if k == 1:
        predicted_index = tf.argmax(pred[0][0][masked_index])
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return predicted_token
    else:
        predicted_indices = tf.nn.top_k(pred[0][0][masked_index], k=k, sorted=True).indices
        predicted_tokens = [tokenizer.convert_ids_to_tokens([predicted_index])[0] for predicted_index in predicted_indices]
        return predicted_tokens
