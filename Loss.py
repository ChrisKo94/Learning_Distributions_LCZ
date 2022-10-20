import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss

def KL_Distr(y_true, y_pred):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    # KL formula
    alpha_pred = tf.math.exp(y_pred)
    # Scale predicted logits (HARD CODED)
    alpha_pred = tf.math.multiply(tf.math.divide(alpha_pred,
                                                 tf.reshape(tf.repeat(np.sum(alpha_pred, axis=1), repeats=y_true.shape[1]),
                                                            shape=[-1,y_true.shape[1]])),
                                  18.5)
    alphas_prior = y_true
    comp_1 = tf.math.lgamma(tf.math.reduce_sum(alpha_pred, 1, keepdims=True))
    comp_2 = -tf.math.reduce_sum(tf.math.lgamma(alpha_pred), 1, keepdims=True)
    comp_3 = -tf.math.lgamma(tf.math.reduce_sum(alphas_prior, 1, keepdims=True))
    comp_4 = tf.math.reduce_sum(tf.math.lgamma(alphas_prior), 1, keepdims=True)
    comp_5 = tf.math.reduce_sum(
        tf.math.multiply(tf.subtract(alpha_pred, alphas_prior),
             tf.subtract(tf.math.digamma(alpha_pred),
                         tf.repeat(tf.math.digamma(tf.math.reduce_sum(alpha_pred, 1, keepdims=True)),
                                   repeats=y_true.shape[1],axis=1)
                         )
             ),
        1, keepdims=True)
    return tf.math.add_n([comp_1,comp_2,comp_3,comp_4,comp_5])

