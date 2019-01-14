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
from sklearn.linear_model import LogisticRegression

class EM:

    def __init__(self, worker_x, influencer_x, annotation_matrix, infl2worker_label, worker2influencer_label,
                 label_set,true_labels):
        self.worker_x = worker_x
        self.influencer_x = influencer_x
        self.annotation_matrix = annotation_matrix
        self.infl2worker_label = infl2worker_label
        self.worker2influencer_label = worker2influencer_label
        self.label_set = label_set
        self.true_labels= true_labels

    # initialization
    def initProbabilities(self):
        # initialize probability z_i (item's quality) randomly
        p_z_i = np.random.randint(2, size=(len(self.infl2worker_label), 1))
        # initialize probability phi_j (worker's reliability) randomly
        p_phi_j = np.random.random((len(self.worker2influencer_label), 1))
        # initialize W_I
        W_I = 2 * np.random.random((influencer_x.shape[1], 1)) - 1
        print(W_I)
        return p_z_i, 1 - p_z_i, p_phi_j, W_I

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def neural_network(self, input_layer):
        theta_i = self.__sigmoid(np.dot(input_layer.values.astype(float), self.W_I))
        return theta_i

    def gradient_ascent(self, x_init, gradient_x, threshold=0.001, eta=0.01):
        x = x_init
        # print (x.shape)
        history = [x]
        done = False
        # while not done:
        gx = gradient_x
        # print (gx.shape)
        x = x + (eta * gx)
        history.append(x)
        # if np.linalg.norm(gx) < threshold:
        #   done = True
        return x, history

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}
        all_named_influencers = self.annotation_matrix[self.annotation_matrix.iloc[:, 2] == 1]
        x_i = self.influencer_x
        self.theta_i = self.neural_network(x_i)
        #self.theta_i = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
        print(self.theta_i)
        p_z_i=self.p_z_i_0.copy().astype(float)
        for infl in range(0, len(infl2worker_label)):
            workers_naming_infl = all_named_influencers[all_named_influencers['influencer'] == infl].worker;
            updated_pz = 1;
            for worker in range(0, workers_naming_infl.shape[0]):
                updated_pz = updated_pz * self.theta_i[infl] * self.p_phi_j[int(workers_naming_infl.iloc[worker])]
                print (self.p_phi_j[int(workers_naming_infl.iloc[worker])])

            p_z_i[infl, 0] = updated_pz
        self.p_z_i_1=p_z_i
        self.p_z_i_0 = 1 - self.p_z_i_1
        return self.p_z_i_0, self.p_z_i_1, self.theta_i

    # M-step

    def Update_phi_wi(self, eta=0.001):
        #update the weights WI
        lr = LogisticRegression(penalty='l2', solver='sag', max_iter=100)
        lr.fit(self.influencer_x.values, self.true_labels.values)
        lr.score(self.influencer_x.values, self.true_labels.values)
        self.W_I= lr.coef_.T

        print("iterations",lr.n_iter_)
        for worker in range(0,len(self.worker2influencer_label)):
            annotation_worker=self.annotation_matrix[self.annotation_matrix['worker']==worker]
            for infl in range(0,len(influencer_x)):
                label_worker_infl=annotation_worker[annotation_worker['influencer']==infl].label
                if label_worker_infl.iloc[0] == 1:
                    grad_phi=((self.p_z_i_1[infl] / self.p_phi_j[worker]) - (self.p_z_i_0[infl] / (1 - self.p_phi_j[worker])))
                else:
                    grad_phi = ((self.p_z_i_0[infl] / self.p_phi_j[worker]) - (self.p_z_i_1[infl] / (1 - self.p_phi_j[worker])))

                 #eq 24 in the document
                self.p_phi_j[worker]= self.p_phi_j[worker] + (eta*grad_phi)

        return self.W_I,self.p_phi_j

    def Run(self, iterr=100):

        self.p_z_i_0, self.p_z_i_1, self.p_phi_j, self.W_I = self.initProbabilities()
        while iterr > 0:
            print(iterr)
            # E-step
            self.p_z_i_0, self.p_z_i_1, theta_i = self.Update_e2lpd()

            # M-step
            self.W_I,self.p_phi_j = self.Update_phi_wi()
            print(self.W_I.shape,self.p_phi_j.shape,self.p_z_i_0.shape,self.p_z_i_1.shape)
            # compute the likelihood
            # print (self.computelikelihood())

            iterr -= 1

        return self.p_z_i_0, self.p_z_i_1,theta_i, self.p_phi_j, self.W_I

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
    datafile = '../input/answers.csv'
    annotationfile = '../output/aij.csv'
    ld = LoadData(datafile)
    worker_x, influencer_x = ld.run_load_data()
    aij, all_workers, all_infl,true_labels = ld.generate_annotation_matrix(datafile)
    annotation_matrix = pd.DataFrame(data=aij, columns=['worker', 'influencer', 'label'])
    annotation_matrix.to_csv(annotationfile, sep=",", index=False)
    infl2worker_label, worker2influencer_label, label_set = get_w2il_i2wl(annotationfile)
    p_z_i_0, p_z_i_1,theta_i, p_phi_j, weight = EM(worker_x, influencer_x, annotation_matrix, infl2worker_label,
                                           worker2influencer_label, label_set,true_labels).Run()
    answers = pd.read_csv(datafile)
    print(pd.DataFrame(data=np.concatenate([all_infl.reshape(all_infl.shape[0], 1), p_z_i_0,p_z_i_1], axis=1),
                       columns=['influencer', 'p0','p1']))
