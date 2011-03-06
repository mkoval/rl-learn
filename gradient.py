#!/usr/bin/env python

from copy import deepcopy
from math import exp
from numpy import array, zeros
from random import gauss, uniform
from sys import argv, exit

class Environment:
	def __init__(self, s0, smin, smax):
		self.s = deepcopy(s0)
		self.smin = smin
		self.smax = smax

	def GetState(self):
		return deepcopy(self.s)

	def DoAction(self, a):
		snew   = self.s + a
		snew   = min(max(snew, self.smin), self.smax)
		a      = snew - self.s
		r      = -(self.s ** 2) - (a ** 2)
		self.s = snew
		return (a, r, snew)

class Policy:
	def __init__(self, w0):
		self.w = deepcopy(w0)

	def GetParams(self):
		return deepcopy(self.w)

	def SetParams(self, w):
		self.w = deepcopy(w)

	def GetDims(self):
		return len(self.w)

	def GetEligibility(self, s, a):
		mu    = self.w[0] * s
		sigma = 1 / (1 + exp(-self.w[1]))

		e       = zeros(len(self.w))
		e_mu    = (a - mu) / sigma ** 2
		e_sigma = ((a - mu) ** 2 - sigma ** 2) / sigma ** 3

		e[0] = e_mu * s
		e[1] = e_sigma * exp(self.w[1]) / (exp(self.w[1]) + 1) ** 2
		return e

	def ChooseAction(self, s):
		mu    = self.w[0] * s
		sigma = 1 / (1 + exp(-self.w[1]))
		return gauss(mu, sigma)

def learn_sga(world, policy, tmax, alpha, gamma, b):
	s = world.GetState();
	w = policy.GetParams()
	D = zeros(policy.GetDims())

	for t in range(0, tmax):
		# Select an action using the policy and perform it on the world.
		a = policy.ChooseAction(s)
		a, r, s = world.DoAction(a)

		# Move in along the gradient of the expected reward function.
		e = policy.GetEligibility(s, a)
		D = e + gamma * D
		w = w + alpha * (1 - gamma) * (r - b) * D
		policy.SetParams(w)

		print('w1 = {0}, w2 = {1}'.format(w[0], w[1]))

def main(args):	
	# SGA Parameters
	alpha    = 0.01
	gamma    = 0.90
	baseline = 0.00
	steps    = 5001

	# World and Policy Parameters
	state_min = -4.0
	state_max = +4.0
	state     = uniform(state_min, state_max)
	weights   = array([ 0.35 + uniform(-0.15, +0.15), 0.00 ])

	world  = Environment(state, state_min, state_max)
	policy = Policy(weights)
	learn_sga(world, policy, steps, alpha, gamma, baseline)

if __name__ == '__main__':
	exit(main(argv))
