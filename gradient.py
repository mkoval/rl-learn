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
		self.s = snew
		r      = -pow(snew, 2) - pow(a, 2)
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
		k     = len(self.w)
		mu    = self.w[0] * s
		sigma = 1 / (1 + exp(-self.w[1]))
		e     = zeros(k)
		e[0]  = (a - mu) * s
		e[1]  = (pow(a - mu, 2) - pow(sigma, 2)) * (1 - sigma)
		return e

	def ChooseAction(self, s):
		mu    = self.w[0] * s
		sigma = 1 / (1 + exp(-self.w[1]))
		return gauss(mu, sigma)

def learn_sga(world, policy, tmax, alpha, gamma, b):
	s = [ None ] * (tmax + 1)
	a = [ None ] * (tmax + 1)
	r = [ 0.0 ]  * (tmax + 1)

	D = zeros(policy.GetDims())
	W = policy.GetParams()
	s[0] = world.GetState()

	for t in range(0, tmax):
		# Select an action using the policy and perform it on the world.
		a[t] = policy.ChooseAction(s[t])
		a[t], r[t], s[t + 1] = world.DoAction(a[t])

		# Move in along the gradient of the expected reward function.
		e = policy.GetEligibility(s[t], a[t])
		D = e + gamma * D
		W = W + alpha * (1 - gamma) * (r[t] - b) * D
		policy.SetParams(W)

def main(args):	
	# SGA Parameters
	alpha    = 0.01
	gamma    = 0.90
	baseline = 0.00
	steps    = 1001

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
