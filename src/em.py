#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:21:41 2018

@author: inesarous
"""

from load_data import LoadData
import pandas as pd
import csv
import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


class EM:

    def __init__(self, influencer_x, annotation_matrix, infl2worker_label, worker2influencer_label,
                 label_set, true_labels):
        self.influencer_x = influencer_x
        self.annotation_matrix = annotation_matrix
        self.infl2worker_label = infl2worker_label
        self.worker2influencer_label = worker2influencer_label
        self.label_set = label_set
        self.true_labels = true_labels

    # initialization
    def init_probabilities(self):
        # initialize probability z_i (item's quality) randomly
        p_z_i = np.random.randint(2, size=(len(self.infl2worker_label), 1)).astype(float)
        # initialize probability phi_j (worker's reliability) randomly
        p_phi_j = 0.5*np.ones((len(self.worker2influencer_label), 1))
        print "p_phi_j_init",p_phi_j[:10]
        return p_z_i, 1 - p_z_i, p_phi_j

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}
        #all_named_influencers = self.annotation_matrix[self.annotation_matrix.iloc[:, 2] == 1]
        print len(self.influencer_x)
        for infl in range(0, len(self.infl2worker_label)):
            updated_pz_1 = 1
            updated_pz_0 = 1
            infl_aij = self.annotation_matrix[self.annotation_matrix['influencer'] == infl]
            T_i = infl_aij[infl_aij['label'] == 1].worker.values
            for worker in T_i.astype(int):
                updated_pz_1 = updated_pz_1 * self.theta_i[infl] * self.p_phi_j[worker]
                updated_pz_0 = updated_pz_0 * (1 - self.theta_i[infl]) * (
                            1 - self.p_phi_j[worker])
            print("upz1", updated_pz_1[:10], "upz0", updated_pz_0[:10])
            self.p_z_i_1[infl] = updated_pz_1 * 1.0 / (updated_pz_0 + updated_pz_1)
            print updated_pz_1 * 1.0 / (updated_pz_0 + updated_pz_1)
            self.p_z_i_0[infl] = updated_pz_0 * 1.0 / (updated_pz_0 + updated_pz_1)
        print "pz0=", self.p_z_i_0[:10], "\npz1=", self.p_z_i_1[:10]
        return self.p_z_i_0, self.p_z_i_1, self.theta_i

    # M-step

    def Update_phi_wi(self, classifier, eta=0.001):
        # update theta_i

        # Fitting the data to the training dataset
        prob_e_step = self.p_z_i_0
        y = np.concatenate((self.true_labels[0:50], prob_e_step[50:]))
        classifier.fit(self.influencer_x, y, epochs=10, verbose=2)
        eval_model = classifier.evaluate(self.influencer_x, y)

        print(eval_model)
        print ("weights", classifier.get_weights())
        self.theta_i = classifier.predict(self.influencer_x)
        print("theta", self.theta_i[:10])
        for worker in range(0, len(self.worker2influencer_label)):
            annotation_worker = self.annotation_matrix[self.annotation_matrix['worker'] == worker]
            for infl in range(0, len(influencer_x)):
                label_worker_infl = annotation_worker[annotation_worker['influencer'] == infl].label
                if label_worker_infl.iloc[0] == 1:
                    grad_phi = ((self.p_z_i_1[infl] / self.p_phi_j[worker]) - (
                                self.p_z_i_0[infl] / (1 - self.p_phi_j[worker])))
                else:
                    grad_phi = ((self.p_z_i_0[infl] / self.p_phi_j[worker]) - (
                                self.p_z_i_1[infl] / (1 - self.p_phi_j[worker])))

                # eq 24 in the document
                self.p_phi_j[worker] = self.p_phi_j[worker] + (eta * grad_phi)

        return self.theta_i, self.p_phi_j, classifier

    def run(self, iterr=20):
        self.p_z_i_0, self.p_z_i_1, self.p_phi_j = self.init_probabilities()
        # initialization
        #self.p_phi_j = np.array([[0.1], [0.8], [0.4], [0.7], [0.6], [0.8], [0.8], [0.3], [0.6], [0.4], [0.7]])
        #self.p_z_i_0 = np.array([[1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0]])
        #self.p_z_i_1 = 1 - self.p_z_i_0
        classifier = Sequential()
        # First Hidden Layer
        layer0 = Dense(3, activation='sigmoid', kernel_initializer='random_normal',
                       input_dim=self.influencer_x.shape[1])
        classifier.add(layer0)
        # Output Layer
        layer1 = Dense(1, activation='sigmoid', kernel_initializer='random_normal')
        classifier.add(layer1)
        # Compiling the neural network
        classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        y = np.concatenate((self.true_labels[0:50], self.p_z_i_1[50:]))
        print self.influencer_x.shape,y.shape
        classifier.fit(self.influencer_x, y, epochs=10, verbose=2)
        self.theta_i = classifier.predict(self.influencer_x)
        print("theta", self.theta_i[:10])
        # self.theta_i=np.array([[0.61],[0.51],[0.613],[0.581 ],[0.61 ],[0.61],[0.58 ],[0.61]])
        while iterr > 0:
            # print(iterr)
            # E-step
            self.p_z_i_0, self.p_z_i_1, theta_i = self.Update_e2lpd()

            # M-step
            self.theta_i, self.p_phi_j, classifier = self.Update_phi_wi(classifier)

            # compute the likelihood
            # print (self.computelikelihood())
            iterr -= 1

        return self.p_z_i_0, self.p_z_i_1, self.theta_i, self.p_phi_j, classifier

    def computelikelihood(self):

        lh_1 = (self.p_z_i_0 * np.log(1 - self.p_phi_j)) + (self.p_z_i_1 * np.log(self.p_phi_j))
        lh_2 = (self.p_z_i_0 * np.log(1 - self.theta_i)) + (self.p_z_i_1 * np.log(self.theta_i))

        lh = lh_1 + lh_2

        return lh


def get_w2il_i2wl(datafile):
    infl2worker_label = {}
    worker2influencer_label = {}
    label_set = []

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        worker, influencer, label = line
        if influencer not in infl2worker_label:
            infl2worker_label[influencer] = []

        infl2worker_label[influencer].append([worker, label])
        if worker not in worker2influencer_label:
            worker2influencer_label[worker] = []
        worker2influencer_label[worker].append([influencer, label])
        if label not in label_set:
            label_set.append(label)
    return infl2worker_label, worker2influencer_label, label_set


if __name__ == '__main__':
    datafile = '../input/fashion_crowd.csv'
    annotationfile = '../output/aij.csv'
    # ld = LoadData(datafile)
    # worker_x, influencer_x = ld.run_load_data()
    # aij, all_workers, all_infl = ld.generate_annotation_matrix_vem(datafile)
    # annotation_matrix = pd.DataFrame(data=aij, columns=['worker', 'influencer', 'label'])
    # annotation_matrix.to_csv(annotationfile, sep=",", index=False)
    # influencer_x.to_csv('../output/influencer_x.csv', sep=",", index=False, encoding='utf-8')
    # worker_x.to_csv('../output/worker_x.csv', sep=",", index=False, encoding='utf-8')
    # all_infl.to_csv('../input/labels_fashion_infl_1.csv', sep=",", index=False)

    annotation_matrix = pd.read_csv('../input/aij_labeled.csv', sep=",")
    influencer_x = pd.read_csv('../output/influencer_x.csv', sep=",", encoding='utf-8')
    true_labels = pd.read_csv('../input/labels_fashion_infl_1.csv', sep=",")
    label_nan = true_labels.dropna()
    true_labels = label_nan[['label']].values
    indices = label_nan.iloc[:, 0]
    ind = indices.tolist()
    labeled_influencer_x=influencer_x.iloc[ind, :]

    worker_x = pd.read_csv('../output/worker_x.csv', sep=",", encoding='utf-8')
    #
    # aij = np.zeros((all_workers.shape[0] * labeled_influencer_x.shape[0], 3))
    # i=0
    # for infl in ind:
    #     infl_aij = annotation_matrix[annotation_matrix['influencer'] == infl]
    #     aij[i:i+len(worker_x)]=infl_aij
    #     i=i+len(worker_x)
    # np.savetxt('../input/aij_labeled.csv', aij)
    infl2worker_label, worker2influencer_label, label_set = get_w2il_i2wl('../input/aij_labeled.csv')

    labeled_influencer_x = labeled_influencer_x.drop(['user_name'], axis=1)
    p_z_i_0, p_z_i_1, theta_i, p_phi_j, weight = EM(labeled_influencer_x, annotation_matrix, infl2worker_label,
                                                       worker2influencer_label, label_set, true_labels).run()
    answers = pd.read_csv(datafile)
    print(pd.DataFrame(data=np.concatenate([all_infl.reshape(all_infl.shape[0], 1), np.where(p_z_i_0 > 0.5, 0, 1),
                                           true_labels.values.reshape(all_infl.shape[0], 1)], axis=1),
                      columns=['influencer', 'classification', 'truth']))
