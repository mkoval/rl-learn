import numpy

class FiniteDifferenceEstimator:
	def __init__(self, world, policy, distance):
		self.world  = world
		self.policy   = policy
		self.distance = distance
	
	def Rollout(self, s, w, t_max):
		R = 0
		for t in range(0, t_max):
			a = self.policy.ChooseAction(w, s)
			s, a, r = self.world.DoAction(s, a)
			R += r
		return R / t_max

class ForwardDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy, distance):
		FiniteDifferenceEstimator.__init__(self, world, policy, distance)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = world.SeedState()
		Un = self.Rollout(w, s, t_max) 

		for i in range(0, len(dw)):
			Up = self.Rollout(s, w + dw[:, i] * self.distance, t_max)
			G[i] = (Up - Un) / self.distance
		return G

class BackwardDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy, distance):
		FiniteDifferenceEstimator.__init__(self, world, policy, distance)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = world.SeedState()
		Up = self.Rollout(w, s, t_max) 

		for i in range(0, len(dw)):
			Un = self.Rollout(s, w - dw[:, i] * self.distance, t_max)
			G[i] = (Up - Un) / self.distance
		return G

class CentralDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy, distance):
		FiniteDifferenceEstimator.__init__(self, world, policy, distance)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = self.world.SeedState()

		for i in range(0, len(dw)):
			Up = self.Rollout(s, w + dw[:, i] * self.distance, t_max)
			Un = self.Rollout(s, w - dw[:, i] * self.distance, t_max)
			G[i] = (Up - Un) / (self.distance * 2)
		return G

