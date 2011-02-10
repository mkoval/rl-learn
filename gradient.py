import random

class WorldLQR:
	def __init__(self, x0, stddev):
		self.x      = min(max(x0, self.minval), self.maxval)
		self.stddev = stddev
		self.minval = -4.0
		self.maxval = +4.0
	
	def GetState(self):
		return self.x

	def DoAction(self, action):
		reward = -pow(self.x, 2) - pow(action, 2)
		noise  = random.gauss(0, self.stddev)
		self.x = self.x + action + noise
		self.x = min(max(self.x, self.minval), self.maxval)
		return reward
