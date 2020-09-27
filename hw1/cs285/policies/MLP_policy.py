import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

<<<<<<< HEAD
<<<<<<< HEAD
        # TODO return the action that the policy prescribes
        action = self(obs)
        return ptu.to_numpy(action)
=======
=======
<<<<<<< HEAD
>>>>>>> rev 1
=======
>>>>>>> 71c761843058117b418f69c07d302874734ece60
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9
        # ODO return the action that the policy prescribes
        if self.discrete:
            output = self(observation)
            _, action = torch.max(output)
        else:
            action = self(observation)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9
        return action
>>>>>>> rev 1
=======
        return action.sample()
>>>>>>> fuck
=======
<<<<<<< HEAD
        return action
>>>>>>> rev 1
=======
        return action.sample()
>>>>>>> fuck
=======
        return action.sample()
>>>>>>> 71c761843058117b418f69c07d302874734ece60
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    #TODO
    def forward(self, observation: torch.FloatTensor) -> Any:
<<<<<<< HEAD
<<<<<<< HEAD
        mean = self.mean_net(ptu.from_numpy(observation))
        dist = torch.distributions.Normal(mean, torch.exp(self.logstd))
        return dist.rsample()
=======
        net = self.mean_net if not self.discrete else self.logits_na
<<<<<<< HEAD
        return net(observation)
>>>>>>> rev 1
=======
        dist = torch.distributions.Normal(net(torch.Tensor(observation)), self.logstd)
        dist.requires_grad = True
        return dist
>>>>>>> fuck
=======
        net = self.mean_net if not self.discrete else self.logits_na
<<<<<<< HEAD
<<<<<<< HEAD
        return net(observation)
>>>>>>> rev 1
=======
        dist = torch.distributions.Normal(net(torch.Tensor(observation)), self.logstd)
        dist.requires_grad = True
        return dist
>>>>>>> fuck
=======
        dist = torch.distributions.Normal(net(torch.Tensor(observation)), self.logstd)
        dist.requires_grad = True
        return dist
>>>>>>> 71c761843058117b418f69c07d302874734ece60
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9
        self.optimizer.zero_grad()
        sampled = self(observations)
        actions = ptu.from_numpy(actions)
        loss = self.loss(actions, sampled)
        loss.backward()
        self.optimizer.step()
=======
>>>>>>> rev 1
=======
=======
<<<<<<< HEAD
>>>>>>> fuck
=======
>>>>>>> 71c761843058117b418f69c07d302874734ece60
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9
        self.optimizer.zero_grad()
        actions = ptu.from_numpy(actions)
        sampled = self(observations).sample()
        sampled.requires_grad = True
        loss = self.loss(actions, sampled)
        loss.backward()
        self.optimizer.step()

<<<<<<< HEAD
>>>>>>> fuck
=======
<<<<<<< HEAD
>>>>>>> rev 1
=======
>>>>>>> fuck
=======
>>>>>>> 71c761843058117b418f69c07d302874734ece60
>>>>>>> 29e4f2aa00c4db48daf78e04b61430483c0542a9

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
