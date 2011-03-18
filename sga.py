#!/usr/bin/env python

from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy
import math
import matplotlib
import matplotlib.pyplot as pyplot
import random

def constrain(x, x_min, x_max):
	return max(min(x, x_max), x_min)

def ChooseAction(w, s):
	mu    = w[0] * s
	sigma = 1 / (1 + math.exp(-w[1]))
	return random.gauss(mu, sigma)

def PerformAction(s0, a0):
	s1 = max(min(s0 + a0, +4), -4)
	a1 = s1 - s0
	r1 = -pow(s0, 2) - pow(a1, 2)
	return (s1, a1, r1)

def Rollout(w, s, t_max):
	R = 0
	for t in range(0, t_max):
		a       = ChooseAction(w, s)
		s, a, r = PerformAction(s, a)
		R       = R + r
	return R / t_max

def Gradient(w, s, epsilon, t_max):
	k  = len(w)
	dw = numpy.eye(k, dtype=float)
	G  = numpy.zeros([ k ], dtype=float)

	for i in range(0, k):
		Up = Rollout(w + dw[:, i] * epsilon, s, t_max) 
		Un = Rollout(w - dw[:, i] * epsilon, s, t_max)
		G[i] = (Up - Un) / (2 * epsilon)
	
	return G / k

def EstimateUtility(w_min, w_max, w_num, s, t_max, epsilon):
	k  = len(w_min)
	i1 = range(0, w_num[0])
	i2 = range(0, w_num[1])
	w1 = numpy.linspace(w_min[0], w_max[0], w_num[0])
	w2 = numpy.linspace(w_min[1], w_max[1], w_num[1])

	U = numpy.empty(w_num, dtype=float)
	G = numpy.zeros([ w_num[0], w_num[1], 2 ], dtype=float)

	for i in itertools.product(i1, i2):
		w = numpy.empty([ 2 ], dtype=float)
		(w[0], w[1]) = (w1[i[0]], w2[i[1]])
		U[i[1], i[0]]    = Rollout(w, s, t_max)
		G[i[1], i[0], :] = Gradient(w, s, epsilon, t_max)

	return (U, G)

def SGA(w, s0, alpha, epsilon, t_max, steps):
	U = numpy.empty(steps, dtype=float)

	for i in range(0, steps):
		G = Gradient(w, s0, epsilon, t_max)
		U[i] = Rollout(w, s0, t_max)
		w = w + alpha * G
	
	return (w, U)

def main():
	s_mu    = 0.0
	s_sigma = 1e-3

	w_min = [ -5, -100 ]
	w_max = [ +5, +100 ]
	w_num = [ 25,   25 ]

	t_max   = 25
	num     = 25
	epsilon = 1.0

	w1 = numpy.linspace(w_min[0], w_max[0], w_num[0])
	w2 = numpy.linspace(w_min[1], w_max[1], w_num[1])
	U = numpy.zeros(w_num, dtype=float)
	G = numpy.zeros([ w_num[0], w_num[1], 2 ], dtype=float)
	W1, W2 = numpy.meshgrid(w1, w2)

	"""
	for i in range(0, num):
		s = 0.15 + random.uniform(-0.15, +0.15)
		print('Simulating {0}/{1} start states'.format(i + 1, num))
		dU, dG = EstimateUtility(w_min, w_max, w_num, s, t_max, epsilon)
		U += dU
		G += dG

	# Three-Dimensional Surface
	figure  = pyplot.figure()
	axis    = figure.gca(projection='3d')
	surface = axis.plot_surface(W1, W2, numpy.transpose(U), cmap=matplotlib.cm.jet, rstride=1, cstride=1)

	# Gradient Vector Field
	figure  = pyplot.figure()
	axis    = figure.gca()
	quiver  = axis.quiver(W1, W2, G[:, :, 0], G[:, :, 1])
	pyplot.show()
	"""

	Ut = numpy.zeros(1000, dtype=float)
	for i in range(0, 100):
		s0 = 0.15 + random.uniform(-0.15, +0.15)
		w0 = numpy.array([ 0.35, 0.00 ])
		w_max, U_sample = SGA(w0, s0, 0.001, epsilon, t_max, 1000)
		Ut += U_sample
		print('{0}/{1}, <w1, w2> = <{2}, {3}>'.format(i + 1, 100, w_max[0], w_max[1]))

	Ut = Ut / 100

	figure  = pyplot.figure()
	axis    = figure.gca()
	quiver  = axis.plot(range(0, 1000), Ut)
	pyplot.show()

	print(w_max)

if __name__ == '__main__':
	main()
