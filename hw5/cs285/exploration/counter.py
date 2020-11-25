from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch, numpy as np


class StateCounter:
	def __init__(self, beta=2):
		self.counts = {}
		self.beta = beta

	def count_state(self, obs: np.ndarray):
		idx = self.quantize_state(obs)
		counts = []
		for index in idx:
			count = self.counts.get(index, 0)
			if count:
				self.counts[index] += 1
			else:
				self.counts[index] = 1
			counts.append(count)
		counts = np.asarray(counts)
		return self.beta / np.sqrt(counts)

	def quantize_state(self, obs):
		obs = obs.round(3)
		a = obs[:, 0].astype(str)
		b = obs[:, 1].astype(str)
		idx = np.core.defchararray.add(a, b)
		return idx


