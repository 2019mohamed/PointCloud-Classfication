# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:09:32 2021

@author: M
"""

import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover, Entropy , Eccentricity,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#data, _ = datasets.make_circles(n_samples=2000, noise=0.05, factor=0.3, random_state=42)
#print(data.shape)
#plot_point_cloud(data)
#data = np.random.randint(0,100,(1000,3))
#filter_func = Projection(columns=[0, 1])
#filter_func = Eccentricity()
def Mappe (data , intervals):
    filter_func = Eccentricity()
    cover = CubicalCover(n_intervals=intervals, overlap_frac=0.3)
    clusterer = DBSCAN()    
    n_jobs = 2    
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
    )
    g = pipe.fit_transform(data)
    A = g.get_edgelist()
    G = nx.Graph(A)
    return G
    #print(G.number_of_nodes(),' ',G.number_of_edges())
    #nx.draw_networkx(G)
    #plt.show()