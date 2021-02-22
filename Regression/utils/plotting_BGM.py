import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import linalg
from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    fig = plt.figure(figsize=(8, 10))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=20, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(np.amin(X[:, 0]) - np.ptp(X[:, 0]*0.05),
             np.amax(X[:, 0]) + np.ptp(X[:, 0]*0.05))
    plt.ylim(np.amin(X[:, 1]) - np.ptp(X[:, 1]*0.05),
             np.amax(X[:, 1]) + np.ptp(X[:, 1]*0.05))
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
