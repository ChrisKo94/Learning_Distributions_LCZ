import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

# Todo: cross-check with jakob's code
# Todo: Add mahalanobis, spectral angle, mse and others if needed
'''
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
'''

def dirichlet_kl_divergence(alpha_c_target, alpha_c_pred, eps=10e-10):

    alpha_c_pred = tf.math.exp(alpha_c_pred)
    alpha_0_target = tf.reduce_sum(alpha_c_target, axis=-1, keepdims=True)
    alpha_0_pred = tf.reduce_sum(alpha_c_pred, axis=-1, keepdims=True)

    term1 = tf.math.lgamma(alpha_0_target) - tf.math.lgamma(alpha_0_pred)
    term2 = tf.math.lgamma(alpha_c_pred + eps) - tf.math.lgamma(alpha_c_target + eps)

    term3_tmp = tf.math.digamma(alpha_c_target + eps) - tf.math.digamma(alpha_0_target + eps)
    term3 = (alpha_c_target - alpha_c_pred) * term3_tmp

    result = tf.squeeze(term1 + tf.reduce_sum(term2 + term3, keepdims=True, axis=-1))

    return result

def mahala_dist(m, n):
    diff = m - n
    cov = tfp.stats.covariance(tf.transpose(n))
    mull = K.dot(tf.linalg.inv(cov), diff)
    mull2 = K.dot(mull, tf.transpose(diff))
    dist = tf.sqrt(mull2)
    return dist