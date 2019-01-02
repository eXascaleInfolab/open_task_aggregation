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
class EM:

    def __init__(self, worker_x,influencer_x,annotation_matrix,infl2worker_label,worker2influencer_label,label_set):
        self.worker_x = worker_x
        self.influencer_x = influencer_x
        self.annotation_matrix = annotation_matrix
        self.infl2worker_label = infl2worker_label
        self.worker2influencer_label = worker2influencer_label
        self.label_set=label_set
        #initialize W_I
        self.synaptic_weights = 2 * np.random.random((influencer_x.shape[1], 1)) - 1

    # initialization
    def Init_p_z_i(self):
        # initialize probability z_i (item's quality) randomly
        rseed=1
        p_z_i = {}
        self.rs = np.random.RandomState(rseed)
        for infl in self.infl2worker_label:
            p_z_i[infl] = self.rs.rand()
        return p_z_i

    def Init_p_phi_j(self):
        #initialize probability phi_j (worker's reliability) randomly
        # rseed=1
        # p_phi_j = np.zeros((len(worker2influencer_label),1))
        # self.rs = np.random.RandomState(rseed)
        # for w in self.worker2influencer_label:
        p_phi_j=np.random.random((len(self.worker2influencer_label), 1))
        return p_phi_j

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def neural_network(self,input_layer):
        theta_i=self.__sigmoid(np.dot(input_layer.values.astype(float), self.synaptic_weights))
        return theta_i

    def gradient_ascent(x_init,gradient_x,threshold=0.001,eta=0.01):
        x = x_init
        history = [x]
        done = False
        while not done:
            gx = gradient_x
            x += eta * gx
            history.append(x)
            if np.linalg.norm(gx) < threshold:
                done = True
        return x, history

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}
        x_i=self.influencer_x
        self.theta_i=self.neural_network(x_i)
        p_z_i_0=(1-self.p_phi_j)*(1-self.theta_i)
        p_z_i_1=(self.p_phi_j)*self.theta_i
        return p_z_i_0,p_z_i_1,self.theta_i

    # M-step

    def Update_phi_wi(self,eta):
        #x=?
        grad_phi=(x/self.p_phi_j)-((1-x)/(1-self.p_phi_j))
        #grad_wi=?
        self.p_phi_j=gradient_ascent(self.p_phi_j,grad_phi)
        self.synaptic_weights=gradient_ascent(self.synaptic_weights,grad_wi)
        return self.p_phi_j,self.synaptic_weights





    def Run(self, iterr=20):

        self.p_z_i_0 = self.Init_p_z_i()
        self.p_z_i_1 = self.Init_p_z_i()
        self.p_phi_j = self.Init_p_phi_j()
        while iterr > 0:
            # E-step
            self.p_z_i_0, self.p_z_i_1,theta_i = self.Update_e2lpd()

            # M-step
            self.p_phi_j, self.synaptic_weights= self.Update_phi_wi(self.a_ij, p_zi)

            # compute the likelihood
            print self.computelikelihood()

            iterr -= 1

        return self.p_z_i_0,self.p_z_i_1, self.p_phi_j, self.synaptic_weights,theta_i

    def computelikelihood(self):


        lh_1=(self.p_z_i_0*np.log(1-self.p_phi_j))+(self.p_z_i_1*np.log(self.p_phi_j))
        lh_2=(self.p_z_i_0*np.log(1-self.theta_i))+(self.p_z_i_1*np.log(self.theta_i))

        lh =lh_1+lh_2

        return lh


def get_w2il_i2wl(datafile):
    infl2worker_label = {}
    worker2influencer_label = {}
    label_set = []

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        worker,influencer, label = line
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
    annotationfile='../output/aij.csv'
    ld = LoadData(datafile)
    worker_x, influencer_x = ld.run_load_data()
    annotation_matrix = pd.DataFrame(data=ld.generate_annotation_matrix(datafile),columns=['worker','influencer','label'])
    annotation_matrix.to_csv(annotationfile,sep=",",index=False)
    infl2worker_label, worker2influencer_label, label_set= get_w2il_i2wl(annotationfile)
    p_z_i_0,p_z_i_1, p_phi_j,theta_i= EM(worker_x,influencer_x,annotation_matrix,infl2worker_label,worker2influencer_label,label_set).Run()
