#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/5/30
"""

import numpy as np


class BernoulliBandit(object):
  """
  经典的 Bernoulli 多臂老虎机
  """
  def __init__(self, K, seed=42):
    np.random.seed(seed)
    self.probs = np.random.uniform(size=K)
    self.best_idx = np.argmax(self.probs)
    self.best_prob = self.probs[self.best_idx]
    self.K = K

    print(f"随机生成一个 {K} 臂的伯努利老虎机")
    print(f"老虎机获奖概率: {[round(i, 4) for i in self.probs]}")
    print(f"获奖概率最大的拉杆是 {self.best_idx} 号，相应的获奖概率是 {round(self.best_prob, 4)}")

  def step(self, k):
    if np.random.rand() < self.probs[k]:
      return 1
    else:
      return 0


def main():
  seed = 42
  K = 10
  bb = BernoulliBandit(K, seed)
  print(f"随机生成一个 {K} 臂的伯努利老虎机")
  print(f"获奖概率最大的拉杆是 {bb.best_idx} 号，相应的获奖概率是 {bb.best_prob}")


if __name__ == '__main__':
  main()
