import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Encoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, n_heads, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.supports_masking = True

        self.attention = layers.MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        self.norm_1 = layers.LayerNormalization()
        self.norm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )

        proj_input = self.norm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.norm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "n_heads": self.n_heads,
            }
        )

        return config
