#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:42:44 2019

@author: zhanglemei and peng
"""

from numpy.linalg import solve
from sklearn.metrics import mean_squared_error, recall_score
import numpy as np
import pathlib
import matplotlib.pyplot as plt


class ExplicitMF:
    def __init__(self,
                 ratings,
                 n_factors=40,
                 threshold_recommendation=0.1,
                 learning_alg='sgd',
                 item_reg=0.0,
                 user_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=True):
        """
        Initialize corresponding params.
        
        Params:
            ratings: (2D array) user x item matrix with corresponding ratings.
            n_factors: (int) number of latent factors after matrix factorization.
            learning_alg (string) name of the optimizer (sgd or als)
            iterm_reg: (float) Regularization term for item latent factors.
            user_reg: (float) Regularization term for user latent factors.
            item_bias_reg (float) Bias of the Regularization term for item
            user_bias_reg (float) Bias of the Regularization term for user
            verbose: (bool) Whether or not to print out training progress.
        """
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.threshold_recommendation = threshold_recommendation
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning_alg
        self._v = verbose

        if self.learning == "sgd":
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)

        self.predictions = []

    def getParams(self):
        return self.n_factors, self.learning, self.item_reg, self.user_reg
        
    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """
        Alternating Least Squares for training process.
        
        Params:
            latent_vectors: (2D array) vectors need to be adjusted.
            fixed_vecs: (2D array) vectors fixed.
            ratings: (2D array) rating matrx.
            _lambda: (float) regularization coefficient. 
        """
        if type == 'user':
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda
            
            for u in range(latent_vectors.shape[0]):
                latent_vectors[u,:] = solve((YTY + lambdaI),
                              ratings[u,:].dot(fixed_vecs))
        elif type == 'item':
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i,:] = solve((XTX + lambdaI),
                              ratings[:,i].T.dot(fixed_vecs))
                
        return latent_vectors
    
    def train(self, n_iter=10, learning_rate=0.1):
        # initialize latent vectors for training process
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        if self.learning == "sgd":
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])

            # Train steps
            self.partial_train(n_iter)
        elif self.learning == "als":
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print(f'\tcurrent iteration: {ctr}')
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.ratings, self.user_reg,
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.ratings, self.item_reg,
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)  # error

            # Update biases
            user_biases_update = (e - self.user_bias_reg * self.user_bias[u])
            self.user_bias[u] += self.learning_rate * user_biases_update
            item_biases_update = (e - self.item_bias_reg * self.item_bias[i])
            self.item_bias[i] += self.learning_rate * item_biases_update

            # Update latent factors
            user_update = (e * self.item_vecs[i, :] - self.user_reg * self.user_vecs[u, :])
            self.user_vecs[u, :] += self.learning_rate * user_update
            item_update = (e * self.user_vecs[u, :] - self.item_reg * self.item_vecs[i, :])
            self.item_vecs[i, :] += self.learning_rate * item_update

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T) + (self.global_bias + self.user_bias[u] + self.item_bias[i])

    def predict_all(self):
        # Initialize predictions
        predictions = np.zeros((self.user_vecs.shape[0], self.item_vecs.shape[0]))

        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                prediction = self.predict(u, i)
                predictions[u, i] = prediction
        return predictions

    def make_recommendations(self,raw_predictions, user_idk, num_recommendation):
        # Turn the predicted score into a binary value by using a threshold (default = 0.1)
        predictions = np.where(raw_predictions >= self.threshold_recommendation, 1, 0)

        # Retrieve user ratings and predicted ratings
        user_ratings = self.ratings[user_idk]
        user_predictions = predictions[user_idk]

        # Evaluate predictions (recall)
        recall = recall_score(user_ratings, user_predictions)

        # Make predictions
        unknown_items = np.argwhere(user_ratings == 0)
        user_new_item_predictions = predictions[user_idk, unknown_items]

        # Retrieve and sort the items recommended (indexes of the columns)
        recommended_items = np.argsort(user_new_item_predictions, axis=0)[::-1]

        # Make recommendation: select the top k items
        recommendations = recommended_items[:num_recommendation]
        return recommendations, recall

    def get_mse(self, pred, actual):
        """Calculate mean squard error between actual ratings and predictions"""
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    def get_recall(self, pred, actual):
        predictions = np.where(pred.flatten() >= self.threshold_recommendation, 1, 0)
        actual = actual.flatten()
        return recall_score(pred, actual)

    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE during train and test iterations.
        
        Params:
            iter_array: (list) List of numbers of iterations to train for each step of 
                        the learning curve.
            test: (2D array) test dataset.
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0

        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print(f"\nIteration: {n_iter}")
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)
            
            predictions = self.predict_all()
            
            self.train_mse += [self.get_mse(predictions, self.ratings)]
            self.test_mse += [self.get_mse(predictions, test)]

            if self._v:
                print(f"Train MSE: {round(self.train_mse[-1], 4)}")
                print(f"Test MSE: {round(self.test_mse[-1], 4)}")
            iter_diff = n_iter

        return predictions, self.test_mse[-1]

    def plot_learning_curve(self, iter_array, model):
        pathlib.Path("plots").mkdir(exist_ok=True)

        plt.suptitle("Collaborative Filtering")  # , fontsize=22
        plt.title("Explicit Matrix Factorization (MF)")  # , fontsize=15, pad=-10
        plt.plot(iter_array, model.train_mse, label='Training', linewidth=5)
        plt.plot(iter_array, model.test_mse, label='Test', linewidth=5)

        # plt.xlim(left=0)
        # plt.margins(3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('MSE', fontsize=15)
        plt.legend(loc='best', fontsize=14)
        plt.savefig("./plots/MatrixFactorization(CF).png")
        plt.show()
