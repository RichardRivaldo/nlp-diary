import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from .decoder import Decoder
from .encoder import Encoder
from .positional_embedding import PositionalEmbedding


class Translation:
    def __init__(
        self,
        model_path="app/translation/assets/translation.h5",
        in_vect_path="app/translation/assets/id_vectorizer.pkl",
        out_vect_path="app/translation/assets/en_vectorizer.pkl",
    ) -> None:

        # Load saved model and vectorizers
        tf_classes = {
            "Encoder": Encoder,
            "Decoder": Decoder,
            "PositionalEmbedding": PositionalEmbedding,
        }
        self.transformer = self.load_transformer(tf_classes, model_path)
        self.in_vect = self.load_vectorizer(in_vect_path)
        self.out_vect = self.load_vectorizer(out_vect_path)

        self.sos = "thisissos"
        self.eos = "thisiseos"

    def load_transformer(self, tf_classes, path):
        return keras.models.load_model(path, custom_objects=tf_classes)

    def load_vectorizer(self, path):
        v = pickle.load(open(path, "rb"))
        vec = TextVectorization.from_config(v["config"])
        vec.set_vocabulary(v["vocab"])
        return vec

    def translate(self, input_sentence, max_seq_len=222):
        output_vocab = self.out_vect.get_vocabulary()
        output_lookup = dict(zip(range(len(output_vocab)), output_vocab))

        tokenized_input = self.in_vect([input_sentence])
        output = self.sos

        for i in range(max_seq_len):
            tokenized_target = self.out_vect([output])[:, :-1]
            predictions = self.transformer([tokenized_input, tokenized_target])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = output_lookup[sampled_token_index]
            output += " " + sampled_token

            if sampled_token == self.eos:
                break

        output = output.replace(f"{self.sos} ", "")
        output = output.replace(f" {self.eos}", "")

        return output
