import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def pca_reduction(dataset, n_components: int, copy: bool):
    pca = PCA(n_components=n_components, copy=copy).fit_transform(dataset)

    return pca


def tsne_reduction(dataset, n_components: int, n_iter: int):
    tsne = TSNE(n_components=n_components,
                n_iter=n_iter).fit_transform(dataset)
    return tsne
