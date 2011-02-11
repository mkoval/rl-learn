#!/usr/bin/env python

import math, random, sys

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
		self.w1 = weights[0]
		self.w2 = weights[1]
	
	def Update(self, weights):
		self.w1 = weights[0]
		self.w2 = weights[1]

	def GetParams(self):
		return 2
	
	def GetPartial(self, i, state, action):
		mu    = self.w1 * state
		sigma = 1.0 / (1 + math.exp(-self.w2))

		if i == 0:
			return (action - mu) * state
		else:
			return (pow(action - mu, 2) - pow(sigma, 2)) * (1 - sigma)
	
	def ChooseAction(self, state):
		mu    = self.w1 * state
		sigma = 1.0 / (1 + math.exp(-self.w2))
		return random.gauss(mu, sigma)

def LearnSGA(world, policy, alpha, gamma, baseline, tmax):
	state   = [ 0.0 ] * tmax
	reward  = [ 0.0 ] * tmax
	action  = [ 0.0 ] * tmax
	D = [ [ 0.0 ] * tmax for i in range(0, policy.GetParams()) ]
	e = [ 0.0 ] * 2
	w = [ 0.0 ] * 2

	state[0] = world.GetState()

	for t in range(1, tmax):
		action[t - 1]           = policy.ChooseAction(state[t - 1])
		state[t], reward[t - 1] = world.DoAction(action[t - 1])

		D[i][t] = [ 0.0 ] * 2

		for i in range(0, policy.GetParams()):
			e[i]    = policy.GetPartial(i, state[t], action[t - 1])
			D[i][t] = e[i] + gamma * D[i][t - 1]
			delta_w = (reward[t - 1] - baseline) * D[i][t]
			w[i]   += alpha * (1 - gamma) * delta_w

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

	start  = random.uniform(minval, maxval)
	world  = LQRWorld(minval, maxval, start, stddev)
	policy = LQRPolicy([ init_w1, init_w2 ])
	LearnSGA(world, policy, alpha, gamma, baseline, tmax)

if __name__ == '__main__':
	sys.exit(main(sys.argv))
