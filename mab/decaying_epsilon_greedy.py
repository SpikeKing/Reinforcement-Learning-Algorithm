#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/5/30
"""
import numpy as np

from solver import Solver


class DecayingEpsilonGreedy(Solver):
  """
  随着时间衰减的 Epsilon-Greedy 算法
  """
  def __init__(self, bandit, init_prob=1.0):
    super(DecayingEpsilonGreedy, self).__init__(bandit)
    self.estimates = np.array([init_prob] * self.bandit.K)
    self.total_count = 0

  def run_one_step(self):
    self.total_count += 1
    if np.random.random() < 1 / self.total_count:
      k = np.random.randint(0, self.bandit.K)
    else:
      k = np.argmax(self.estimates)

    r = self.bandit.step(k)
    self.estimates[k] += 1 / (self.counts[k] + 1) * (r - self.estimates[k])

    return k