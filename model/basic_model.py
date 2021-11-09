from __future__ import absolute_import, print_function, unicode_literals, division

import tensorflow as tf
import tensorflow_hub as hub


def fallback_backbone():
    return hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5",
                          trainable=False)


class SubModel(tf.keras.Model):
    def __init__(self, code_length=32, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.code_length = code_length
        if not pretrained:
            raise NotImplementedError

        self.bbn = tf.keras.layers.Dense(code_length)
        self.cbn = tf.keras.layers.Dense(512)


class BasicModel(tf.keras.Model):
    def __init__(self, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.preamble = fallback_backbone() if pretrained else tf.identity




