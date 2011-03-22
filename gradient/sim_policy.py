#!/usr/bin/env python

import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as pyplot
import numpy
import random

from FDEstimator import *
from LQRWorld    import *

# Parameters for estimating the expected reward function.
rollouts = 200
t_max    = 10
epsilon  = 0.1

# Define the continuous MDP.
world     = LQRWorld(0.0, 0.5)
policy    = LQRPolicy()
estimator = CentralDifferenceEstimator(world, policy, epsilon)

# Region over which to estimate the expected reward function.
w1_range = numpy.linspace(-2,  +2,  20)
w2_range = numpy.linspace(-10, +10, 10)
w_range  = [ w1_range, w2_range ]

w_size    = map(len, w_range)
grad_size = w_size + [ policy.GetDims() ]

G = numpy.zeros(grad_size)
R = numpy.zeros(w_size)

for i1 in range(0, w_size[0]):
	for i2 in range(0, w_size[1]):
		for j in range(0, rollouts):
			w = numpy.array([ w1_range[i1], w2_range[i2] ])
			s = world.SeedState()
			R[i1, i2]    += estimator.Rollout(s, w, t_max) / rollouts
			G[i1, i2, :] += estimator.EstimateGradient(w, t_max) / rollouts

# Expected Average Reward
W1, W2  = numpy.meshgrid(w1_range, w2_range)
figure  = pyplot.figure()
axis    = figure.gca(projection='3d')
surface = axis.plot_surface(W1, W2, numpy.transpose(R), cmap=matplotlib.cm.jet,
                            rstride=1, cstride=1)

axis.set_xlabel('w1')
axis.set_ylabel('w2')
axis.set_zlabel('Average Reward')

# Estimated Gradient
W1, W2  = numpy.meshgrid(w1_range, w2_range)
figure  = pyplot.figure()
axis    = figure.gca()
quiver  = axis.quiver(W1, W2, G[:,:,0], G[:,:,1])

axis.set_xlabel('w1')
axis.set_ylabel('w2')

pyplot.show()
