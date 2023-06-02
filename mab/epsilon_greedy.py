#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/5/30
"""
import numpy as np

from solver import Solver


class EpsilonGreedy(Solver):
  """
  Epsilon Greedy 算法
  """
  def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
    super(EpsilonGreedy, self).__init__(bandit)

    self.epsilon = epsilon
    # 初始化全部拉杆的期望，后续逐渐更新
    self.estimates = np.array([init_prob] * self.bandit.K)

  def run_one_step(self):
    if np.random.random() < self.epsilon:
      k = np.random.randint(0, self.bandit.K)   # 随机选择一个拉杆
    else:
      k = np.argmax(self.estimates)   # 选择最大期望的拉杆
    r = self.bandit.step(k)
    # 更新期望，先run，再更新count，所以count+1，更新期望
    self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
    return k
