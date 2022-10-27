import numpy as np
from tensorflow import keras
from transformers import BertTokenizerFast, TFBertMainLayer

class Detection:
    def __init__(
        self,
        model_path="app/detection/assets/detection.h5",
        max_length=512
    ) -> None:
        tf_classes = {
            "TFBertMainLayer": TFBertMainLayer
        }

        self.model = keras.models.load_model(model_path, custom_objects=tf_classes)
        self.max_length = max_length

    def tokenize(self, texts):
        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            do_lower_case = True
        )

        result = tokenizer(
            text = texts,
            add_special_tokens = True,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'tf'
        )

        return {
            'input_ids': result['input_ids'],
            'attention_mask': result['attention_mask'],
            'token_type_ids': result['token_type_ids']
        }

    def detect(self, input_sentence):
        tokenized_input = self.tokenize([input_sentence])
        pred = np.round(self.model.predict(tokenized_input))
        if pred[0][0] == 0: return "en"
        else: return "id" 
