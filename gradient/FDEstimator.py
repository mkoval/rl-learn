import numpy

class FiniteDifferenceEstimator:
	def __init__(self, world, policy):
		self.world  = world
		self.policy = policy
	
	def Rollout(self, s, w, t_max):
		R = 0
		for t in range(0, t_max):
			a = self.policy.ChooseAction(w, s)
			s, a, r = self.world.DoAction(s, a)
			R += r

		return R / t_max

class ForwardDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy):
		FiniteDifferenceEstimator.__init__(self, world, policy)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = world.SeedState()
		Un = self.Rollout(w, s, t_max) 

		for i in range(0, len(dw)):
			Up = self.Rollout(s, w + dw[:, i] * epsilon, t_max)
			G[i] = (Up - Un) / epsilon

		return G / k

class BackwardDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy):
		FiniteDifferenceEstimator.__init__(self, world, policy)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = world.SeedState()
		Up = self.Rollout(w, s, t_max) 

		for i in range(0, len(dw)):
			Un = self.Rollout(s, w - dw[:, i] * epsilon, t_max)
			G[i] = (Up - Un) / epsilon

		return G / k

class CentralDifferenceEstimator(FiniteDifferenceEstimator):
	def __init__(self, world, policy):
		FiniteDifferenceEstimator.__init__(self, world, policy)

	def EstimateGradient(self, w, t_max):
		dw = numpy.eye(len(w), dtype=float)
		G  = numpy.zeros(len(w), dtype=float)
		s  = world.SeedState()

		for i in range(0, len(dw)):
			Up = self.Rollout(s, w + dw[:, i] * epsilon, t_max)
			Un = self.Rollout(s, w - dw[:, i] * epsilon, t_max)
			G[i] = (Up - Un) / epsilon

		return G / k
