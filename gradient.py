#!/usr/bin/env python

import math, numpy, random, sys

class LQRWorld:
	def __init__(self, minval, maxval, x0, stddev):
		self.x      = min(max(x0, minval), maxval)
		self.stddev = stddev
		self.minval = minval
		self.maxval = maxval

	def GetState(self):
		return self.x

	def DoAction(self, action):
		reward = -pow(self.x, 2) - pow(action, 2)
		noise  = random.gauss(0, self.stddev)
		self.x = self.x + action + noise
		self.x = min(max(self.x, self.minval), self.maxval)
		return (self.x, reward)

class LQRPolicy:
	def __init__(self, weights):
		self.Update(weights)

	def Update(self, weights):
		self.w1 = weights[0]
		self.w2 = weights[1]
	
	def GetDims(self):
		return 2

	def GetWeights(self):
		return numpy.array([ self.w1, self.w2 ])

	def GetParams(self, state):
		 mu    = self.w1 * state
		 sigma = 1.0 / (1 + math.exp(-self.w2))
		 return numpy.array([ mu, sigma ])

	def GetEligibility(self, state, action):
		mu, sigma = self.GetParams(state)
		e1 = (action - mu) * state
		e2 = (pow(action - mu, 2) - pow(sigma, 2)) * (1 - sigma)
		return numpy.array([ e1, e2 ])

	def ChooseAction(self, state):
		mu, sigma = self.GetParams(state)
		return random.gauss(mu, sigma)

def LearnSGA(world, policy, alpha, gamma, baseline, tmax):
	state    = numpy.zeros(tmax + 1)
	reward   = numpy.zeros(tmax + 1)
	action   = numpy.zeros(tmax + 1)
	state[0] = world.GetState()

	k  = policy.GetDims()
	w  = policy.GetWeights()
	d  = numpy.zeros(k)
	e  = numpy.zeros(k)
	dw = numpy.zeros(k)

	for t in range(0, tmax):
		action[t] = policy.ChooseAction(state[t])
		state[t + 1], reward[t] = world.DoAction(action[t])

		e  = policy.GetEligibility(state[t], action[t])
		d  = e + gamma * d
		dw = (reward[t] - baseline) * d
		w  = w + alpha * (1 - gamma) * dw

		mu, sigma = policy.GetParams(state[t])
		print('t = {0: 4}: mu = {1: 5.3f}, sigma = {2: 5.3f}'.format(t, mu, sigma))

		policy.Update(w)

	return policy

def main(args):
	minval   = -4
	maxval   = +4
	tmax     = 5000
	alpha    = 0.01
	gamma    = 0.90
	baseline = 0.00
	stddev   = 0.50
	init_w1  = 0.35 + random.uniform(-0.15, +0.15)
	init_w2  = 0.00
	start    = 1.00

	print('Simulating {0} steps with parameters:'.format(tmax))
	print('  learning rate   = {0: .3f}'.format(alpha))
	print('  discount factor = {0: .3f}'.format(gamma))
	print('  baseline        = {0: .3f}'.format(baseline))
	print('  w1(init)        = {0: .3f}'.format(init_w1))
	print('  w2(init)        = {0: .3f}'.format(init_w2))
	print('  state(init)     = {0: .3f}'.format(start))
	print('')

	world  = LQRWorld(minval, maxval, start, stddev)
	policy = LQRPolicy([ init_w1, init_w2 ])
	LearnSGA(world, policy, alpha, gamma, baseline, tmax)

if __name__ == '__main__':
	sys.exit(main(sys.argv))
