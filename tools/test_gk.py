# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:00:37 2016

@author: jqmviegas

GK Tests
"""

from sklearn import datasets
import skfuzzy as fuzz
import pandas as pd

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first .two features.
Y = iris.target

cntr, u, u0, d, jm, p, fpc, covs = fuzz.cluster.gk(
        X.T, 3, 1.5, 
        error=0.005, maxiter=10000, init=None)

tmp = u.argmax(axis=0)

data = pd.DataFrame(X)

data["cluster"] = tmp

data.plot.scatter(x=0, y=1, c="cluster")


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, 3, 2, 
        error=0.005, maxiter=1000, init=None)

tmp = u.argmax(axis=0)

data = pd.DataFrame(X)

data["cluster"] = tmp

data.plot.scatter(x=0, y=1, c="cluster")

