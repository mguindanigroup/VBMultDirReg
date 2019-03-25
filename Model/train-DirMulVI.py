import os

import tensorflow as tf
# CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
# CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
# sess = tf.Session(config=CONFIG)

import h5py
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import time
from collections import defaultdict
import pandas as pd
import matplotlib as mpl
from utils import supp_func
from sklearn import metrics
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp
import pickle as pkl



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hard_sigmoid(x):
    return np.minimum(np.maximum(x, np.zeros_like(x)), np.ones_like(x))

def hard_sigmoid(x):
    return np.minimum(np.maximum(x, np.zeros_like(x)), np.ones_like(x))

def compute_statistics(data, s, pred, VI='True'):
    """
    Inputs:
        data: dictionary containing xi_true
        s: binary indicator of non-zero beta values
        pred: predicted beta value

    Outputs:
        [precision, recall, MCC, AUC, F1, ACC], [fpr, tpr, roc_auc]
    """
    p = data['X'].shape[1]
    q = data['Y'].shape[1]
    y = data['xi_true'].T.reshape(1, p * q)[0]
    if VI == False:
        y = data['xi_true'].reshape(1, p * q)[0]  # .T needed when compute VI statistics
    pred_bin = s.reshape(1, p * q)[0]
    pred = pred.reshape(1, p * q)[0]

    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, pred_bin).ravel()

    P = 25  # (p+q)*inclusion
    N = 2500 - 25  # p*q - P
    FDR = fp / (fp + tp)
    ACC = (tp + tn) / (P + N)
    MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    AUC = metrics.roc_auc_score(y, pred)
    return [precision, recall, MCC, AUC, F1, ACC], [fpr, tpr, roc_auc]


def compute_implied_fdr(threshold, s):
    """
    Inputs:
        s: predicted probability
        threshold: level greater than threshold are selected

    Outputs:
        fdr corresponding to the threshold
    """
    indicator = np.asarray([s > threshold])
    return np.sum((1 - s) * indicator) / np.sum(indicator)


def search_threshold(s, fdr):
    """
    Inputs:
        s: predicted probability
        fdr: controlled false discovery rate level

    Outputs:
        largest threshold such that the fdr is less than the controlled level: fdr
    """
    for threshold in np.linspace(0, 1, 101):
        if compute_implied_fdr(threshold, s) < fdr:
            break
    return threshold


def plot_roc(fpr, tpr, roc_auc, color, linestyle, method_name):
    mean_auc = np.mean(roc_auc)
    std_auc = np.std(roc_auc)
    mean_tpr = np.mean(tpr, axis=0)
    std_tpr = np.std(tpr, axis=0)
    plt.plot(fpr, mean_tpr, color=color, linestyle=linestyle,
             label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (method_name, mean_auc, std_auc),
             lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(fpr, tprs_lower, tprs_upper, color=color, alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Average False Positive Rate')
    plt.ylabel('Average True Positive Rate')
    plt.legend(loc="lower right")





def train_model(data, epoch):
    """
    Inputs:
        data: dictionary containing taxa counts (Y) and covariates (X)
        compute: class for computing negative log likelihood and KL distance
        epoch: number of iterations
    Outputs:
        pred: PPI (posterior probability of inclusion)
        s: binarized decision on whether or not include an association
        valpha: log of alpha parameter value in concrete distribution, used for compute PPI
    """
    hp = {'p': data['X'].shape[1],
          'q': data['Y'].shape[1],
          'K': 100,
          'zeta': 1.1,
          'gamma': -0.1,
          'gamma_zeta_logratio': np.log(0.1 / 1.1),
          'beta_p': 0.666
          }

    K = hp['K']
    p = hp['p']
    q = hp['q']
    zeta = hp['zeta']
    gamma = hp['gamma']
    gamma_zeta_logratio = hp['gamma_zeta_logratio']
    beta_p = hp['beta_p']

    tf.reset_default_graph()

    mu_beta_0 = tf.get_variable("mu_beta_0", shape=[p, q], initializer=tf.contrib.layers.xavier_initializer())
    mu_beta_1 = tf.get_variable("mu_beta_1", shape=[p, q], initializer=tf.contrib.layers.xavier_initializer())

    alpha = tf.get_variable("alpha", shape=[p, q], initializer=tf.contrib.layers.xavier_initializer())
    alpha_rep = tf.stack([alpha] * K)
    bias = tf.get_variable("bias", shape=[1, q], initializer=tf.contrib.layers.xavier_initializer())

    x = tf.placeholder(tf.float32, [data['X'].shape[0], data['X'].shape[1]], name='data_x')
    y = tf.placeholder(tf.float32, [data['Y'].shape[0], data['Y'].shape[1]], name='data_y')

    ###########################  define model flow
    u = tf.random_uniform(shape=(K, p, q), minval=0, maxval=1, dtype=tf.float32)
    s0 = tf.sigmoid((tf.math.log(u) - tf.math.log(1 - u) + alpha_rep) / beta_p)
    s = s0 * (zeta - gamma) + gamma  # stretch s
    mask = tf.clip_by_value(s, 0, 1)  # hard_sigmoid(s) shape=(K,p)
    ###########################
    mu_beta = tf.where(tf.random_uniform([p]) - 0.5 < 0, mu_beta_0, mu_beta_1)
    mu_beta_rep = tf.stack([mu_beta] * K)

    noise = tf.random_normal(shape=(K, p, q), mean=0, stddev=1, dtype=tf.float32)
    beta = tf.math.add(noise, mu_beta_rep)

    masked_weights = tf.multiply(mask, beta)  # shape(K,p)
    activation = tf.reduce_mean(tf.tensordot(x, masked_weights, axes=[[1], [1]]),
                                axis=1) + bias  # ?tf.add #shape=(None,100)
    y_pred = tf.exp(activation)
    ##################

    class compute_loss(object):
        def __init__(self, p=None, q=None, K=None):
            self.K = K
            self.p = p
            self.q = q

        def NegDM_loglike(self, labels, y_pred):
            delta = y_pred  # the first ouput corresponds to delta; 2nd corresponds to L0 penalty
            negloglike = -(tf.lgamma(labels + delta) - tf.lgamma(delta))
            negloglike = tf.reduce_sum(negloglike, axis=-1) \
                         - (tf.lgamma(tf.reduce_sum(delta, axis=-1)) - tf.lgamma(
                tf.reduce_sum(labels, axis=-1) + tf.reduce_sum(delta, axis=-1)))
            negloglike = tf.reduce_sum(negloglike)
            return negloglike

        def KL_welling(self, beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio):
            log_qz = tf.reduce_sum(tf.math.log(tf.sigmoid(alpha - beta_p * gamma_zeta_logratio)))
            qz = tf.reduce_sum(tf.sigmoid(alpha - beta_p * gamma_zeta_logratio))
            log_pz = tf.reduce_sum(tf.math.log(np.zeros([self.K, self.p, self.q], dtype=np.float32) + 0.5)) / self.K
            KLD = log_qz - log_pz + tf.reduce_sum(
                -0.5 * tf.square(mu_beta) + tf.reduce_mean(tf.math.log(tf.square(beta)),
                                                           axis=0))  # - log_pbeta + log_qbeta
            return qz, KLD

        def KL_MC_nonlocal(self, beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio):
            qz = tf.sigmoid(alpha - beta_p * gamma_zeta_logratio)
            neg_entropy_q = tf.reduce_mean(
                tf.log(0.5 / np.sqrt(2 * np.pi) * tf.math.exp(-0.5 * tf.square(beta - mu_beta_0)) \
                       + 0.5 / np.sqrt(2 * np.pi) * tf.math.exp(-0.5 * tf.square(beta - mu_beta_1))), axis=0)
            KLD_tau_2 = tf.reduce_sum(qz * tf.log(qz / 0.5) + (1 - qz) * tf.log((1 - qz) / 0.5)) + \
                        tf.reduce_sum(qz * (neg_entropy_q + \
                                            0.125 * tf.square(mu_beta_0) + 0.125 * tf.square(mu_beta_1) + \
                                            np.log(np.sqrt(2 * np.pi)) + 2 * np.log(2) - \
                                            tf.reduce_mean(tf.math.log(1 * tf.square(beta) + 1e-10), axis=0)))
            return KLD_tau_2, neg_entropy_q, qz  # log_qz, log_qbeta, log_pz, log_pbeta, KLD

    compute = compute_loss()
    Neglikelihood = compute.NegDM_loglike(y, y_pred)
    KLD_tau_2, neg_entropy_q, qz = compute.KL_MC_nonlocal(beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio)
    loss = Neglikelihood + 1 * KLD_tau_2
    ##################

    lr = tf.placeholder(tf.float32, name='learning_rate')
    train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[mu_beta_1, mu_beta_0, alpha, bias])

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(epoch):
        _, cost, vKL, vNeglikelihood, valpha, vmu_beta, vneg_entropy_q, vqz = sess.run(
            [train_op1, loss, KLD_tau_2, Neglikelihood, alpha, mu_beta, neg_entropy_q, qz], \
            feed_dict={x: data['X'], y: data['Y'], lr: 0.001})  #
        if i % 4000 == 0:
            pred = sigmoid(valpha) * (zeta - gamma) + gamma
            print('epoch', i)
    beta = 2 / 3
    zeta = 1.1
    gamma = -0.1
    s = sigmoid(valpha) * (zeta - gamma) + gamma
    s = hard_sigmoid(s)
    threshold = search_threshold(s, 0.1)
    s[s < threshold] = 0
    s[s > threshold] = 1

    pred = sigmoid(valpha) * (zeta - gamma) + gamma
    pred = hard_sigmoid(pred)
    print('threshold: {} and implied FDR: {}'.format(threshold, compute_implied_fdr(threshold, pred)) )
    return s, valpha, pred

if __name__ == '__main__':
    repetition = 50
    data_dir = "/Users/luyadong/PycharmProjects/DMVI/data/"
    p = 50
    q = 50
    pred_matrix = np.zeros([repetition, p, q])
    statistics_matrix = np.zeros([repetition, 6])

    with h5py.File(data_dir + 'rep_50_001.h5', 'r') as f:
        cov_duncan_01 = np.transpose(np.asarray(f['cov_duncan_001']), axes=[2, 1, 0])
        response_duncan_01 = np.transpose(np.asarray(f['response_duncan_001']), axes=[2, 1, 0])
        truebeta_duncan_01 = np.transpose(np.asarray(f['truebeta_duncan_001']), axes=[2, 1, 0])
    ind = 20
    XX = cov_duncan_01[ind, :, 1:]
    YY = response_duncan_01[ind, :, :]
    truebeta = truebeta_duncan_01[ind, :, :]
    # data = {}
    # data['X'] = XX
    # data['Y'] = YY
    truebeta = truebeta_duncan_01[ind, :, :]
    xi_true = np.zeros_like(truebeta)
    xi_true[truebeta != .0] = 1
    # data['xi_true'] = xi_true
    # plt.imshow(xi_true)

    for ind in range(repetition):
        print('start repetition {}'.format(ind))
        ### prepare data
        XX = cov_duncan_01[ind, :, 1:]
        YY = response_duncan_01[ind, :, :]
        truebeta = truebeta_duncan_01[ind, :, :]
        data = {}
        data['X'] = XX
        data['Y'] = YY
        data['truebeta'] = truebeta
        xi_true = np.zeros_like(truebeta)
        xi_true[truebeta != .0] = 1
        data['xi_true'] = xi_true

        hp = {'p': data['X'].shape[1],
              'q': data['Y'].shape[1],
              'K': 100,
              'zeta': 1.1,
              'gamma': -0.1,
              'gamma_zeta_logratio': np.log(0.1 / 1.1),
              'beta_p': 0.666
              }
        ### train model

        s, valpha, pred = train_model(data, epoch=10000)
        plt.imshow(s)
        plt.savefig('xi_pred.png')

        print('finish training')
        print('mean and max PPI', np.mean(pred), np.max(pred))
        pred_matrix[ind, :, :] = pred
        ### compute statistics
        statistics_matrix[ind, :], [fpr, tpr, roc_auc] = compute_statistics(data, s, pred, VI='True')
        print('[precision, recall, MCC, AUC, F1, ACC]: \n', statistics_matrix[ind, :])