
import os

import tensorflow as tf
# CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
# CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
# sess = tf.Session(config=CONFIG)

import h5py
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from utils import supp_func
from utils import RealDataExp
import pickle as pkl


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hard_sigmoid(x):
    return np.minimum(np.maximum(x, np.zeros_like(x)), np.ones_like(x))

def compute_implied_fdr(threshold, s):
    indicator = np.asarray( [s>threshold] )
    return np.sum( (1-s)*indicator ) / np.sum(indicator)

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
          'K': 200,
          'zeta': 1.1,
          'gamma': -0.1,
          'gamma_zeta_logratio': np.log(0.1 / 1.1),
          'beta_p': 0.666
          }
    print(hp)

    K = hp['K']
    p = hp['p']
    q = hp['q']
    zeta = hp['zeta']
    gamma = hp['gamma']
    gamma_zeta_logratio = hp['gamma_zeta_logratio']
    beta_p = hp['beta_p']

    tf.reset_default_graph()
    tf.set_random_seed(1234)
    mu_beta_0 = tf.get_variable("mu_beta_0", shape=[p, q], \
                                #                             initializer = tf.initializers.random_uniform(-1,1))
                                initializer=tf.contrib.layers.xavier_initializer())
    mu_beta_1 = tf.get_variable("mu_beta_1", shape=[p, q], \
                                #                             initializer = tf.initializers.random_uniform(-1,1))
                                initializer=tf.contrib.layers.xavier_initializer())

    alpha = tf.get_variable("alpha", shape=[p, q], initializer=tf.contrib.layers.xavier_initializer())
    alpha_rep = tf.stack([alpha] * K)
    bias = tf.get_variable("bias", shape=[1, q], initializer=tf.contrib.layers.xavier_initializer())

    x = tf.placeholder(tf.float32, [data['X'].shape[0], data['X'].shape[1]], name='data_x')
    y = tf.placeholder(tf.float32, [data['Y'].shape[0], data['Y'].shape[1]], name='data_y')

    ###########################  define the mask (Bernoulli random variable)
    u = tf.random_uniform(shape=(K, p, q), minval=0, maxval=1, dtype=tf.float32)
    s0 = tf.sigmoid((tf.math.log(u) - tf.math.log(1 - u) + alpha_rep) / beta_p)
    s = s0 * (zeta - gamma) + gamma  # stretch s
    mask = tf.clip_by_value(s, 0, 1)  # hard_sigmoid(s) shape=(K,p)
    ###########################
    mu_beta = tf.where(tf.random_uniform([p]) - 0.5 < 0, mu_beta_0, mu_beta_1)
    print('mu_beta shape is ', mu_beta.shape)
    mu_beta_rep = tf.stack([mu_beta] * K)

    noise = tf.random_normal(shape=(K, p, q), mean=0, stddev=1, dtype=tf.float32)
    beta = tf.math.add(noise, mu_beta_rep)

    masked_weights = tf.multiply(mask, beta)  # shape(K,p)
    print('masked_weights/x shape', masked_weights.shape, x.shape)

    activation = tf.reduce_mean(tf.tensordot(x, masked_weights, axes=[[1], [1]]), axis=1) + bias
    print('activation shape', activation.shape)
    y_pred = tf.exp(activation)

    print('y_pred shape', y_pred.shape)

    class compute_loss:
        def __init__(self, p=None, q=None, K=None):
            self.K = K
            self.p = p
            self.q = q
            print('class')

        def NegDM_loglike(self, labels, y_pred):
            delta = y_pred  # the first ouput corresponds to delta; 2nd corresponds to L0 penalty
            negloglike = -(tf.lgamma(labels + delta) - tf.lgamma(delta))
            negloglike = tf.reduce_sum(negloglike, axis=-1) \
                         - (tf.lgamma(tf.reduce_sum(delta, axis=-1)) - tf.lgamma(
                tf.reduce_sum(labels, axis=-1) + tf.reduce_sum(delta, axis=-1)))
            negloglike = tf.reduce_sum(negloglike)
            print('neg shape', negloglike.shape)
            return negloglike

        def KL_welling(self, beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio):
            log_qz = tf.reduce_sum(tf.math.log(tf.sigmoid(alpha - beta_p * gamma_zeta_logratio)))
            qz = tf.reduce_sum(tf.sigmoid(alpha - beta_p * gamma_zeta_logratio))
            log_pz = tf.reduce_sum(tf.math.log(np.zeros([self.K, self.p, self.q], dtype=np.float32) + 0.5)) / self.K
            KLD = log_qz - log_pz + tf.reduce_sum(
                -0.5 * tf.square(mu_beta) + tf.reduce_mean(tf.math.log(tf.square(beta)),
                                                           axis=0))  # - log_pbeta + log_qbeta
            return qz, KLD

        def KL_MC(self, beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio):
            qz = tf.sigmoid(alpha - beta_p * gamma_zeta_logratio)
            neg_entropy_q = tf.reduce_mean(
                tf.log(0.5 / np.sqrt(2 * np.pi) * tf.math.exp(-0.5 * tf.square(beta - mu_beta_0)) \
                       + 0.5 / np.sqrt(2 * np.pi) * tf.math.exp(-0.5 * tf.square(beta - mu_beta_1))), axis=0)
            KLD_tau_2 = tf.reduce_sum(qz * tf.log(qz / 0.5) + (1 - qz) * tf.log((1 - qz) / 0.5)) + \
                        tf.reduce_sum(qz * (neg_entropy_q + \
                                            0.125 * tf.square(mu_beta_0) + 0.125 * tf.square(mu_beta_1) + \
                                            np.log(np.sqrt(2 * np.pi)) + 2 * np.log(2) - \
                                            tf.reduce_mean(tf.math.log(1 * tf.square(beta) + 1e-10), axis=0)))
            return KLD_tau_2, neg_entropy_q, qz

            ################## compute the loss (objective)

    compute = compute_loss(p=hp['p'], q=hp['q'], K=hp['K'])
    Neglikelihood = compute.NegDM_loglike(y, y_pred)
    KLD_tau_2, neg_entropy_q, qz = compute.KL_MC(beta, mu_beta, alpha, noise, beta_p, gamma_zeta_logratio)
    loss = Neglikelihood + 1 * KLD_tau_2
    print('loss shape', Neglikelihood.shape, loss.shape)
    ##################

    lr = tf.placeholder(tf.float32, name='learning_rate')
    train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[mu_beta_1, mu_beta_0, alpha, bias])

    init = tf.global_variables_initializer()
    costs_list = list()

    sess = tf.Session()
    sess.run(init)
    import time
    start_time = time.time()
    for i in range(20000):
        _, cost, vKL, vNeglikelihood, valpha, vmu_beta, vneg_entropy_q, vqz, vbias, vact = sess.run([train_op1,
                                                                                                     loss, KLD_tau_2,
                                                                                                     Neglikelihood,
                                                                                                     alpha, mu_beta,
                                                                                                     neg_entropy_q,
                                                                                                     qz,
                                                                                                     bias,
                                                                                                     activation],
                                                                                                    feed_dict={
                                                                                                        x: data['X'],
                                                                                                        y: data['Y'],
                                                                                                        lr: 0.01})  #
        costs_list.append(cost)
        if i % 1000 == 0:
            s = sigmoid(valpha) * (zeta - gamma) + gamma
            print('epoch', i)
            print('beta', np.min(np.abs(vmu_beta)), np.max(np.abs(vmu_beta)), np.mean(np.abs(vmu_beta)))
            print('bias', np.min(np.abs(vbias)), np.max(np.abs(vbias)), np.mean(np.abs(vbias)))
            print('PPI', np.mean(s), np.max(s))
            print('bacteriod', np.max(s[:, 0]))
            print('loss, KL, Negloglike', cost, vKL, vNeglikelihood)
            print(' neg_entropy_q, qz', np.isnan(vneg_entropy_q).any(), np.isnan(vqz).any(), \
                  np.min(np.abs(vneg_entropy_q)), np.mean(np.abs(vqz)))
    elapsed_time = time.time() - start_time
    print(elapsed_time)
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
    print('implied FDR', compute_implied_fdr(threshold, pred))
    return s, valpha, pred, vmu_beta






if __name__ == '__main__':
    real_data_dir = '/Users/luyadong/PycharmProjects/DMVI/data/'
    data = {}
    data['Y'] = pd.read_csv(real_data_dir + 'adj_taxacount.txt', sep='\t').iloc[:, :]
    data['X'] = pd.read_csv(real_data_dir + 'adj_nutri.txt', sep='\t').iloc[:, 1:]

    print(data['Y'].shape, data['X'].shape)
    assert data['Y'].shape == (98, 30)
    assert data['X'].shape == (98, 117)

    start_time = time.time()
    s, valpha, pred, vmu_beta = train_model(data, epoch=20000)
    with open(real_data_dir + 'pred_matrix_real_data.pkl', 'wb') as file:
        pkl.dump(pred, file)
    elapsed_time = time.time() - start_time
    print(elapsed_time)


    RealDataExp = RealDataExp.RealDataExp(valpha, vmu_beta, data)
    RealDataExp.plot_selected()
    pred, s = RealDataExp.compute_s_pred()
    loc, pos_strength = RealDataExp.compute_selected(s)
    print(pos_strength)
    nutrient_list, taxa_list = RealDataExp.get_nutri_taxa_list(s)
    print(taxa_list)
    G = RealDataExp.draw_bipartite_graph(s, nutrient_list, taxa_list, chenli=False)

