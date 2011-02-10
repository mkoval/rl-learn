import random

class LQRWorld:
	def __init__(self, minval, maxval, x0, stddev):
		self.x      = min(max(x0, self.minval), self.maxval)
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
		return reward

class LQRPolicy:
	def __init__(w1, w2):
		self.w1 = w1
		self.w2 = s2
	
	def Update(self, w1, w2):
		self.w1 = w1
		self.w2 = w2
	
	def ChooseAction(self, state):
		mu    = self.w1 * state
		sigma = 1.0 / (1 + math.exp(-self.w2))
		return random.guass(mu, sigma)

def main(args):
	minval  = -4
	maxval  = +4
	stddev  = 0.50
	init_w1 = 0.35 + random.uniform(-0.15, +0.15)
	init_w2 = 0.00

	start  = random.uniform(minval, maxval)
	world  = LQRWorld(minval, maxval, start, stddev)
	policy = LQRPolicy(init_w1, init_w2)

