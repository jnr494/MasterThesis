# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:08:14 2021

@author: mrgna and https://github.com/imanolperez/market_simulator/blob/master/src/logsig_inversion.py
"""

import numpy as np
from tqdm.auto import tqdm
import copy
import iisignature

def leadlag(path):
    lead = path[1:]
    lag = path[:-1]
    
    leadlag = np.vstack((lead,lag))
    return leadlag

def leadlag_to_path(leadlag_path):
    leadlag_path[1,:]

class Organism:
    def __init__(self, n_points, pip, n_pips):
        self.n_points = n_points
        self.pip = pip
        self.n_pips = n_pips
        
        # Initialise
        self.randomise()

    def __add__(self, other):
        """Breed."""

        derivatives = []
        for derivative1, derivative2 in zip(self.derivatives, other.derivatives):
            if np.random.random() < 0.5:
                derivatives.append(derivative1)
            else:
                derivatives.append(derivative2)

        prices = np.r_[0., np.cumsum(derivatives)]
        path = leadlag(prices)

        o = Organism(self.n_points, self.pip, self.n_pips)
        o.derivatives = derivatives
        o.set_path(path)

        return o

    def random_derivative(self):
        r = self.pip * np.random.randint(-self.n_pips, self.n_pips)

        return r

    def randomise(self):
        self.derivatives = np.array([self.random_derivative() for _ in range(self.n_points - 1)])
        prices = np.r_[0., self.derivatives.cumsum()]
        path = leadlag(prices)
        self.set_path(path)


    def mutate(self, prob=0.1):
        for i in range(len(self.derivatives)):
            if np.random.random() < prob:
                self.derivatives[i] = self.random_derivative()

        prices = np.r_[0., np.cumsum(self.derivatives)]
        path = leadlag(prices)
        self.set_path(path)

    def set_path(self, path):
        self.path = path

    def logsignature(self, order, prep):
        return iisignature.logsig(self.path.T, prep)

    def loss(self, sig, order, prep):
        diff = np.abs((sig - self.logsignature(order, prep)) / sig)
        diff /= 1 + np.arange(len(sig))
        return np.mean(diff)

class Population:
    def __init__(self, n_organisms, n_points, pip, n_pips, d, order):
        self.n_points = n_points
        self.pip = pip
        self.n_pips = n_pips
        self.n_organisms = n_organisms
        
        self.prepare = iisignature.prepare(d, order)
        
        self.organisms = [Organism(n_points, pip, n_pips) for _ in range(n_organisms)]

    def fittest(self, sig, p, order):
        n = int(len(self.organisms) * p)
        return sorted(self.organisms, key=lambda o: o.loss(sig, order, self.prepare))[:n]
        
    def evolve(self, sig, p, order, mutation_prob=0.1):
        parents = self.fittest(sig, p, order)
        new_generation = copy.deepcopy(parents)

        while len(new_generation) != self.n_organisms:
            i = j = 0
            while i == j:
                i, j = np.random.choice(range(len(parents)), size=2)
                parent1, parent2 = parents[i], parents[j]

            child = parent1 + parent2
            child.mutate(prob=mutation_prob)
            
            new_generation.append(copy.deepcopy(child))

        self.organisms = new_generation

        # Return loss
        return new_generation[0].loss(sig, order, self.prepare)

def train(sig, order, n_iterations, n_organisms, n_points, pip, n_pips, d,
          top_p=0.1, mutation_prob=0.1):
    population = Population(n_organisms, n_points, pip, n_pips, d, order)
    pbar = tqdm(range(n_iterations))
    
    count = 0
    for _ in pbar:
        print(count)
        loss = population.evolve(sig, p=top_p, order=order, mutation_prob=mutation_prob)
        pbar.set_description(f"Loss: {loss}")
        pbar.refresh()
        
        count +=1
        
        if loss == 0.:
            break

    return population.fittest(sig, p=top_p, order=order)[0].path, loss #.path[::2, 1]

if __name__ == '__main__':
    import os
    import sys
    import time
    sys.path.insert(1,os.path.dirname(os.getcwd()))

    import BlackScholesModel
    import matplotlib.pyplot as plt
    
    pip, n_pips = (0.01, 500)
    n_iterations, n_organisms = (100,100)
    n = 20
    
    model = BlackScholesModel.BlackScholesModel(1,0.05, 0.3, np.array([[1]]), 0.01, dt = 0.01)

    for _ in range(n):
        model.evolve_s_b()
        
    path = np.squeeze(model.spot_hist)[np.newaxis,:]
    
    plt.plot(path[0])
    plt.show()
    
    leadlag_path = leadlag(path[0])

    for p in leadlag_path:
        plt.plot(p)
    plt.show()
    
    order = 3
    iisignature.logsiglength(2, order)
    prep = iisignature.prepare(2, order)
    logsig = iisignature.logsig(leadlag_path.T,prep)
    
    inverted_logsig = train(logsig, order, n_iterations, n_organisms, n+1, pip, n_pips, 2)
