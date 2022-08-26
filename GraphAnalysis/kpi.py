import networkx as nx
import numpy as np

class FactoryRating():

    def __init__(self, factory, fullPathGraph, PathGraph):

        self.factory = factory
        self.fullPathGraph = fullPathGraph
        self.PathGraph = PathGraph

    def PathWideVariance(self):
        min_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'max_pathwidth').values())))
        return np.mean(min_pathwidth/max_pathwidth)