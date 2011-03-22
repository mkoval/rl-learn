import math
import numpy
import random

class LQRWorld:
	def __init__(self, noise_mu, noise_sigma):
		self.noise_mu    = noise_mu
		self.noise_sigma = noise_sigma
	
	def SeedState(self):
		return random.uniform(-0.15, +0.15)

	def DoAction(self, s0, a):
		s1 = max(min(s0 + a, +4), -4)
		a  = s1 - s0

		a1 = a + random.gauss(self.noise_mu, self.noise_sigma)
		s1 = max(min(s0 + a1, +4), -4)

		r = -s0 ** 2 - a ** 2
		return (s1, a, r)

class LQRPolicy:
	def GetDims(self):
		return 2
	
	def SeedParam(self):
		return numpy.array([ 0.35 + random.uniform(-0.15, +0.15), 0.00 ])

	def GetEligibility(self, w, s, a):
		e = numpy.zeros(len(w), dtype=float)
		mu = w[0] * s
		sigma = 1 / (1 + math.exp(-w[1]))

		# XXX: these may assume alpha is proportional to sigma^2
		e[0] = (a - mu) * s
		e[1] = ((a - mu)**2 - sigma**2) * (1 - sigma)
		return e

	def ChooseAction(self, w, s):
		mu    = w[0] * s
		sigma = 1 / (1 + math.exp(-w[1]))
		return random.gauss(mu, sigma)

