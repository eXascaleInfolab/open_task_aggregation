from load_data import LoadData
import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from math import floor
from sklearn.preprocessing import StandardScaler
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
class EM_VI:
    def __init__(self, worker_x, influencer_x, annotation_matrix, infl2worker_label, worker2influencer_label, label_set,
                 true_labels):
        self.worker_x = worker_x
        self.influencer_x = influencer_x
        self.annotation_matrix = annotation_matrix
        self.infl2worker_label = infl2worker_label
        self.worker2influencer_label = worker2influencer_label
        self.label_set = label_set
        self.true_labels = true_labels
        self.q_z_i_0 = np.random.randint(2, size=(len(self.infl2worker_label), 1)).astype(float)
        self.q_z_i_1 = 1.0 - self.q_z_i_0
        self.A = 8.0
        self.B = 2.0
        self.alpha, self.beta = self.init_alpha_beta()



    def optimize_rj(self,x_train, n_neurons, nb_layers, training_epochs, display_step, batch_size, n_input, alpha, beta):
        keep_prob = tf.placeholder(tf.float64)

        # input layer
        x = tf.placeholder(tf.float64, [None, n_input])
        layer = x
        # hideen layers
        for _ in range(nb_layers):
            layer = tf.layers.dense(inputs=layer, units=n_neurons, activation=tf.nn.tanh)
        # output layer
        alpha_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)
        beta_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)

        dist = tf.distributions.Beta(alpha_prime, beta_prime)
        target_dist = tf.distributions.Beta(alpha, beta)

        loss = tf.distributions.kl_divergence(target_dist, dist)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                avg_cost = 0.0
                total_batch = int(len(x_train) / batch_size)
                x_batches = np.array_split(x_train, total_batch)
                for i in range(total_batch):
                    batch_x = x_batches[i]
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={
                                        x: batch_x,
                                        keep_prob: 0.8
                                    })
                    avg_cost += c / total_batch
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                          "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            alpha_prime_res, beta_prime_res = sess.run([alpha_prime, beta_prime],
                                                       feed_dict={
                                                           x: x_train,
                                                           keep_prob: 0.8
                                                       })
            print("alpha_prime_res=", alpha_prime_res, "beta_prime_res=", beta_prime_res)
            return alpha_prime_res, beta_prime_res


    def init_theta_i(self):
        # learn theta_i
        classifier = Sequential()
        # first hidden layer
        layer0 = Dense(3, activation='sigmoid', kernel_initializer='random_normal',
                       input_dim=self.influencer_x.shape[1])
        classifier.add(layer0)
        # Output Layer
        layer1 = Dense(1, activation='sigmoid', kernel_initializer='random_normal')
        classifier.add(layer1)
        # Compiling the neural network
        classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        y = np.concatenate((self.true_labels[0:int(len(self.infl2worker_label) / 3), None],
                            self.q_z_i_1[int(len(infl2worker_label) / 3):]))
        classifier.fit(self.influencer_x, y, epochs=100)
        theta_i = classifier.predict(self.influencer_x)
        return theta_i, classifier

    def init_alpha_beta(self):
        alpha =  np.zeros((len(worker2influencer_label),1))
        beta =  np.zeros((len(worker2influencer_label),1))
        for w in range(0, len(self.worker2influencer_label)):
            alpha[w] = self.A
            beta[w] = self.B
        return alpha, beta

    def update(self, a, b):
        self.n_update += 1
        self.change += np.abs(a - b)

    def e_step(self, max_it=10):
        for it in range(max_it):
            self.change = 0
            self.n_update = 0
            # update q(z)
            for influencer in range(0, len(infl2worker_label)):
                q_z_i_0 = 1
                q_z_i_1 = 1
                infl_aij=self.annotation_matrix[self.annotation_matrix['influencer'] == influencer]
                T_i=infl_aij[infl_aij['label'] == 1].worker.values
                for worker in T_i.astype(int):
                    alpha = self.alpha[worker]
                    beta = self.beta[worker]
                    q_z_i_0 = q_z_i_0 * (1 - self.theta_i[influencer]) * np.exp(digamma(beta) - digamma(alpha + beta))
                    q_z_i_1 = q_z_i_1 * self.theta_i[influencer] * np.exp(digamma(alpha) - digamma(alpha + beta))
                new_q_z_i_1 = q_z_i_1 * 1.0 / (q_z_i_0 + q_z_i_1)
                self.update(self.q_z_i_1[influencer], new_q_z_i_1)
                self.q_z_i_0[influencer] = q_z_i_0 * 1.0 / (q_z_i_0 + q_z_i_1)
                self.q_z_i_1[influencer] = q_z_i_1 * 1.0 / (q_z_i_0 + q_z_i_1)

            # update q(r)
            new_alpha = np.zeros((len(worker2influencer_label),1))
            new_beta = np.zeros((len(worker2influencer_label),1))
            for worker in range(0, len(self.worker2influencer_label)):
                new_alpha[worker] = self.A
                new_beta[worker] = self.B

            for worker in range(0, len(self.worker2influencer_label)):
                worker_aij=self.annotation_matrix[self.annotation_matrix['worker'] == worker]
                T_j=worker_aij[worker_aij['label'] == 1].influencer.values
                for infl in T_j.astype(int):
                    new_alpha[worker] += (1 - self.theta_i[infl])
                    new_beta[worker] += self.theta_i[infl]
            for worker in range(0, len(self.worker2influencer_label)):
                self.update(self.alpha[worker], new_alpha[worker])
                self.alpha[worker] = new_alpha[worker]
                self.update(self.beta[worker], new_beta[worker])
                self.beta[worker] = new_beta[worker]
            avg_change = self.change * 1.0 / self.n_update
            if avg_change < 0.01:
                break

    def m_step(self, classifier):
        prob_e_step = np.where(self.q_z_i_0 > 0.1, 0, 1)
        print prob_e_step
        y = np.concatenate((self.true_labels[0:int(len(self.infl2worker_label) / 3), None],
                            prob_e_step[int(len(infl2worker_label) / 3):]))
        classifier.fit(self.influencer_x, y, epochs=100)
        self.theta_i = classifier.predict(self.influencer_x)
        n_neurons = 3
        nb_layers = 2
        training_epochs = 100
        display_step = 10
        batch_size = 1
        n_input = self.worker_x.shape[1]
        alpha_prime_res, beta_prime_res = self.optimize_rj(self.worker_x, n_neurons, nb_layers, training_epochs, display_step,
                                                      batch_size, n_input, self.alpha, self.beta)
        self.alpha = alpha_prime_res
        self.beta = beta_prime_res


    def run(self, iterr=10):
        self.theta_i, classifier = self.init_theta_i()
        while iterr > 0:
            # variational E step
            self.e_step()
        #     # variational M step
            self.m_step(classifier)
            iterr -= 1
        return self.q_z_i_0, self.q_z_i_1, self.theta_i, self.alpha, self.beta


def get_w2il_i2wl(datafile):
    infl2worker_label = {}
    worker2influencer_label = {}
    label_set = []
    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        w, infl, lbl = line
        worker = float(w)
        influencer = float(infl)
        label = float(lbl)
        # generate dictionnary for influencers
        if influencer not in infl2worker_label:
            infl2worker_label[influencer] = []
        infl2worker_label[influencer].append([int(worker), int(label)])
        # generate dictionnary for workers
        if worker not in worker2influencer_label:
            worker2influencer_label[worker] = []
        worker2influencer_label[worker].append([int(influencer), int(label)])
        if label not in label_set:
            label_set.append(int(label))
    return infl2worker_label, worker2influencer_label, label_set


if __name__ == '__main__':
    worker_x = pd.read_csv('../input/worker_x.csv')
    influencer_x = pd.read_csv('../input/influencer_x.csv')
    true_labels = pd.read_csv('../input/labels_infl.csv').label
    annotation_matrix =  pd.read_csv('../output/aij.csv')
    # sc = StandardScaler()
    # influencer_x = sc.fit_transform(influencer_x)
    # worker_x = sc.fit_transform(worker_x)
    infl2worker_label, worker2influencer_label, label_set = get_w2il_i2wl('../output/aij.csv')
    qz0, qz1, theta_i, alpha, beta = EM_VI(worker_x, influencer_x, annotation_matrix, infl2worker_label,
                                               worker2influencer_label, label_set, true_labels).run()
    print(pd.DataFrame(data=np.concatenate([ np.where(qz0 > 0.5, 0, 1),
                                            true_labels.values.reshape(true_labels.shape[0], 1)], axis=1),columns=['classification', 'truth']))
