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


def gaussian_regression(x_cords, y_cords, df, title):
    y_cord_1 = df['Cords'].values[0]
    y_cord_2 = df['Cords'].values[1]
    y_cord_3 = df['Cords'].values[2]

    x_lin = tf.linspace(-3.5, 3.5, 200)

    predData1 = train(x_cords[y_cord_1], y_cords[y_cord_1], x_lin)
    predData2 = train(x_cords[y_cord_2], y_cords[y_cord_2], x_lin)
    predData3 = train(x_cords[y_cord_3], y_cords[y_cord_3], x_lin)

    # Plot cluster with applied regression
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_cords[y_cord_1], y_cords[y_cord_1],
                c='navy', label=df['Cluster'].values[0], linewidth=0.5)
    ax1.scatter(x_lin, predData1, c='black',
                label='Fit 1', linewidth=0.1, marker='.')
    ax1.scatter(x_cords[y_cord_2], y_cords[y_cord_2],
                c='c', label=df['Cluster'].values[1], linewidth=0.5)
    ax1.scatter(x_lin, predData2, c='black',
                label='Fit 2', linewidth=0.1, marker='.')
    ax1.scatter(x_cords[y_cord_3], y_cords[y_cord_3],
                c='cornflowerblue', label=df['Cluster'].values[2], linewidth=0.5)
    ax1.scatter(x_lin, predData3, c='black',
                label='Fit 3', linewidth=0.1, marker='.')
    plt.title(title)
    ax1.legend()
    plt.show()
