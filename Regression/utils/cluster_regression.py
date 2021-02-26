import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

from tensorflow import keras


def train(x, y, x_lin):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, activation='elu', input_dim=1))
    model.add(keras.layers.Dense(50, activation='elu'))
    model.add(keras.layers.Dense(50, activation='elu'))

    model.add(keras.layers.Dense(1))

    batch_size = 64
    epochs = 100

    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='logcosh', metrics=['MAE'])
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

    predData = model.predict(x_lin)

    return predData


def gaussian_regression(x_cords, y_cords, y_pred, title):
    y_cord_1 = np.where(y_pred == 0)[0]
    y_cord_2 = np.where(y_pred == 1)[0]
    y_cord_3 = np.where(y_pred == 2)[0]

    x_lin = tf.linspace(-3.5, 3.5, 200)

    # Plot cluster
    '''
    fig, ax = plt.subplots()
    ax.scatter(x_cords[y_cord_1], y_cords[y_cord_1],
               label='Cluster 1', c='navy')
    ax.scatter(x_cords[y_cord_2], y_cords[y_cord_2],
               label='Cluster 2', c='c')
    ax.scatter(x_cords[y_cord_3], y_cords[y_cord_3],
               label='Cluster 3', c='cornflowerblue')
    ax.legend()
    plt.show()
    '''

    predData1 = train(x_cords[y_cord_1], y_cords[y_cord_1], x_lin)
    predData2 = train(x_cords[y_cord_2], y_cords[y_cord_2], x_lin)
    predData3 = train(x_cords[y_cord_3], y_cords[y_cord_3], x_lin)

    # Plot cluster with applied regression
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_cords[y_cord_1], y_cords[y_cord_1],
                c='navy', label='Cluster 1', linewidth=0.5)
    ax1.scatter(x_lin, predData1, c='black',
                label='Fit 1', linewidth=0.1, marker='.')
    ax1.scatter(x_cords[y_cord_2], y_cords[y_cord_2],
                c='c', label='Cluster 2', linewidth=0.5)
    ax1.scatter(x_lin, predData2, c='black',
                label='Fit 2', linewidth=0.1, marker='.')
    ax1.scatter(x_cords[y_cord_3], y_cords[y_cord_3],
                c='cornflowerblue', label='Cluster 3', linewidth=0.5)
    ax1.scatter(x_lin, predData3, c='black',
                label='Fit 3', linewidth=0.1, marker='.')
    plt.title(title)
    ax1.legend()
    plt.show()
