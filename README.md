# nlp-diary

Diary App with NLP Pipelines

## How to Run

### Server

To run the server, just execute `run.sh` in the root directory. Default port used will be `8000`.

## API

### Translation

-   `POST {"input_sentence": "<this_is_your_input>"} /detection` -> Detect your input into English or Indonesian.
-   `POST {"input_sentence": "<this_is_your_input>"} /translation` -> Translate your input into English.

## References

### Language Translation

-   Deep Learning with Python, Second Edition by François Chollet.
-   [English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/)
-   [Neural Machine Translation with TensorFlow](https://blog.paperspace.com/neural-machine-translation-with-tensorflow/)
-   [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)
-   [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

### Language Detection

-   Deep Learning with Python, Second Edition by François Chollet.
-   [How to Train BERT](https://towardsdatascience.com/how-to-train-bert-aaad00533168)
-   [Fine-tuning a BERT model](https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert)
