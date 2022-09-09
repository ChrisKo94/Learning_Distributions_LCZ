from Loss import KL_Distr
import numpy as np
import tensorflow as tf

y_true_ex = np.array([[10.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                      [5.5,5.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                      [3.5,3.5,3.5,1.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])

y_pred_ex = np.array([np.random.normal(size=17),
                     np.random.normal(size=17),
                     np.random.normal(size=17)])

y_pred_ex = y_pred_ex * 10

losses_ex = KL_Distr(y_true_ex, y_pred_ex)