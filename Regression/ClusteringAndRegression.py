#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:24:47 2021

@author: nihal
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from tensorflow.keras.constraints import max_norm

from utils.plotting_BGM import plot_results, plot_boundaries

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

Gaussian_Mixture = True

df = pd.read_csv('data/ex.csv')

dataset = df.copy()
X = dataset.values
x_cords = dataset['x'].values
y_cords = dataset['y'].values


def train(x, y):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, activation='elu', input_dim=1))
    # model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(50, activation='elu'))
    model.add(keras.layers.Dense(50, activation='elu'))

    model.add(keras.layers.Dense(1))

    batch_size = 64
    epochs = 50

    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='logcosh', metrics=['MAE'])
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs)

    predData = model.predict(x)

    return predData


x_train, x_test, y_train, y_test = train_test_split(
    x_cords, y_cords, test_size=0.20, random_state=np.random.seed(65))  # 6

if Gaussian_Mixture:
    X = np.stack((x_test, y_test), axis=1)
    x_cords = x_test
    y_cords = y_test

    gmm = GaussianMixture(
        n_components=3, covariance_type='diag').fit(X)  # full
    y_pred = gmm.predict(X)

    # bgm = BayesianGaussianMixture(n_components=3, covariance_type='diag').fit(X)
    # y_pred = bgm.predict(X)

    y_cord_1 = np.where(y_pred == 0)[0]
    y_cord_2 = np.where(y_pred == 1)[0]
    y_cord_3 = np.where(y_pred == 2)[0]

    fig, ax = plt.subplots()
    ax.scatter(x_cords[y_cord_1], y_cords[y_cord_1],
               label='cluster 1', c='blue')
    ax.scatter(x_cords[y_cord_2], y_cords[y_cord_2],
               label='cluster 2', c='orange')
    ax.scatter(x_cords[y_cord_3], y_cords[y_cord_3],
               label='cluster 3', c='green')
    ax.legend()
    plt.show()

    predData1 = train(x_cords[y_cord_1], y_cords[y_cord_1])
    predData2 = train(x_cords[y_cord_2], y_cords[y_cord_2])
    predData3 = train(x_cords[y_cord_3], y_cords[y_cord_3])

    fig1, ax1 = plt.subplots()
    ax1.scatter(x_cords[y_cord_1], y_cords[y_cord_1],
                c='blue', label='cluster 1', linewidth=0.5)
    ax1.scatter(x_cords[y_cord_1], predData1, c='black',
                label='Fit 1', linewidth=0.5, marker='.')
    ax1.scatter(x_cords[y_cord_2], y_cords[y_cord_2],
                c='orange', label='cluster 2', linewidth=0.5)
    ax1.scatter(x_cords[y_cord_2], predData2, c='black',
                label='Fit 2', linewidth=0.5, marker='.')
    ax1.scatter(x_cords[y_cord_3], y_cords[y_cord_3],
                c='green', label='cluster 3', linewidth=0.5)
    ax1.scatter(x_cords[y_cord_3], predData3, c='black',
                label='Fit 3', linewidth=0.5, marker='.')
    ax1.legend()
    plt.show()

else:
    predData = train(x_train, y_train)
    predData = predData.reshape((predData.shape[0],))

    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, label='Actual Data', c='blue')
    ax.scatter(x_train, predData, label='Prediction', c='Black', linewidth=0.5)
    ax.legend()
    plt.show()

    # #Generate Noise
    # mu, sigma = 0, 1
    # noise = np.random.normal(mu, sigma, len(x_train))
    # noise = 0

    # loc = np.where(np.logical_and(y_train < predData +  noise, y_train > predData - noise))[0]

    # x_train_2 = np.delete(x_train, loc)
    # y_train_2 = np.delete(y_train, loc)

    # predData_2 = train(x_train_2, y_train_2)
    # predData_2 = predData_2.reshape((predData_2.shape[0],))

    # fig, ax = plt.subplots()
    # ax.scatter(x_train_2, y_train_2, label='Actual Data', c='blue')
    # ax.scatter(x_train_2, predData_2, label='Prediction', c='Black', linewidth=0.5)
    # ax.legend()
    # plt.show()
