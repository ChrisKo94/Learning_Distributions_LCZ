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
    diff = tf.math.subtract(m,n)
    corr_mat = np.array([[ 1.        ,  0.1541183 ,  0.13828751,  0.3622234 ,  0.11849896,
        -0.43200201,  0.25036819, -0.25749387,  0.60411027, -0.3355173 ,
         0.02047879, -0.18426484, -0.22348219, -0.29025871,  0.26075699,
        -0.53555426],
       [ 0.1541183 ,  1.        ,  0.62768986,  0.53208411,  0.73824711,
         0.46108267, -0.25573984, -0.17740012, -0.33782624, -0.49809064,
        -0.71365851, -0.67294134, -0.54577035, -0.63223543, -0.62210508,
        -0.3219833 ],
       [ 0.13828751,  0.62768986,  1.        , -0.11090734,  0.51017678,
         0.6406193 , -0.0492585 , -0.10975598, -0.07570037, -0.54338822,
        -0.47458025, -0.63179281, -0.18872664, -0.66387782, -0.35824548,
        -0.48057855],
       [ 0.3622234 ,  0.53208411, -0.11090734,  1.        ,  0.56725374,
         0.03870841, -0.33273862, -0.04821599, -0.29494257, -0.21381497,
        -0.47700836, -0.33735855, -0.48418611, -0.32398513, -0.4303312 ,
        -0.07777664],
       [ 0.11849896,  0.73824711,  0.51017678,  0.56725374,  1.        ,
         0.50669931, -0.15329166, -0.34292036, -0.47304452, -0.59550617,
        -0.68498979, -0.7831226 , -0.15466582, -0.71272533, -0.69158672,
        -0.10233825],
       [-0.43200201,  0.46108267,  0.6406193 ,  0.03870841,  0.50669931,
         1.        , -0.36651356,  0.41797443, -0.54229755, -0.28770592,
        -0.63495527, -0.56819533, -0.23833664, -0.52083768, -0.64967129,
        -0.01699723],
       [ 0.25036819, -0.25573984, -0.0492585 , -0.33273862, -0.15329166,
        -0.36651356,  1.        , -0.26919249,  0.72369426, -0.2802925 ,
        -0.03355334, -0.20997015,  0.15488759,  0.33356174,  0.46383583,
        -0.04315624],
       [-0.25749387, -0.17740012, -0.10975598, -0.04821599, -0.34292036,
         0.41797443, -0.26919249,  1.        ,  0.01448907,  0.19531851,
        -0.1499848 ,  0.07677797, -0.42341645,  0.08437618, -0.01875894,
         0.08961217],
       [ 0.60411027, -0.33782624, -0.07570037, -0.29494257, -0.47304452,
        -0.54229755,  0.72369426,  0.01448907,  1.        , -0.13724641,
         0.21818359,  0.08044623, -0.10509625,  0.29686667,  0.71000396,
        -0.37031642],
       [-0.3355173 , -0.49809064, -0.54338822, -0.21381497, -0.59550617,
        -0.28770592, -0.2802925 ,  0.19531851, -0.13724641,  1.        ,
         0.65809452,  0.63004894,  0.2005491 ,  0.3117738 ,  0.02980497,
         0.4726835 ],
       [ 0.02047879, -0.71365851, -0.47458025, -0.47700836, -0.68498979,
        -0.63495527, -0.03355334, -0.1499848 ,  0.21818359,  0.65809452,
         1.        ,  0.85665601,  0.52885127,  0.36028832,  0.43829117,
         0.12131476],
       [-0.18426484, -0.67294134, -0.63179281, -0.33735855, -0.7831226 ,
        -0.56819533, -0.20997015,  0.07677797,  0.08044623,  0.63004894,
         0.85665601,  1.        ,  0.34987262,  0.61702927,  0.49060938,
         0.09756274],
       [-0.22348219, -0.54577035, -0.18872664, -0.48418611, -0.15466582,
        -0.23833664,  0.15488759, -0.42341645, -0.10509625,  0.2005491 ,
         0.52885127,  0.34987262,  1.        ,  0.16985567,  0.17275122,
         0.26665525],
       [-0.29025871, -0.63223543, -0.66387782, -0.32398513, -0.71272533,
        -0.52083768,  0.33356174,  0.08437618,  0.29686667,  0.3117738 ,
         0.36028832,  0.61702927,  0.16985567,  1.        ,  0.72780108,
         0.16657644],
       [ 0.26075699, -0.62210508, -0.35824548, -0.4303312 , -0.69158672,
        -0.64967129,  0.46383583, -0.01875894,  0.71000396,  0.02980497,
         0.43829117,  0.49060938,  0.17275122,  0.72780108,  1.        ,
        -0.17493192],
       [-0.53555426, -0.3219833 , -0.48057855, -0.07777664, -0.10233825,
        -0.01699723, -0.04315624,  0.08961217, -0.37031642,  0.4726835 ,
         0.12131476,  0.09756274,  0.26665525,  0.16657644, -0.17493192,
         1.        ]]).astype(np.float32)
    corr_mat = tf.convert_to_tensor(corr_mat)
    mull = tf.tensordot(diff, tf.linalg.inv(corr_mat), axes=1)
    mull2 = tf.tensordot(mull,tf.transpose(diff),axes=1)
    diag = tf.linalg.diag_part(mull2)
    dists = tf.math.sqrt(diag)
    return tf.reduce_mean(dists)