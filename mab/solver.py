#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/5/30
"""
import numpy as np


class Solver(object):
  """
  MAB 求解的基类
  """
  def __init__(self, bandit):
    self.bandit = bandit
    self.counts = np.zeros(self.bandit.K)  # 记录动作选择数量
    self.regret = 0.  # 累计懊悔值
    self.actions = []  # 记录每一步的动作
    self.regrets = []  # 记录每一步的懊悔

  def update_regret(self, k):
    """
    更新懊悔值
    """
    self.regret += self.bandit.best_prob - self.bandit.probs[k]  # 概率懊悔值
    self.regrets.append(self.regret)  # 记录概率懊悔值

  def run_one_step(self):
    """
    返回拉动哪一根拉杆
    1. 根据策略选择动作
    2. 根据动作获得奖励
    3. 更新期望奖励估值
    """
    raise NotImplementedError

  def run(self, num_steps):  # 执行
    for _ in range(num_steps):
      k = self.run_one_step()  # 动作是k
      self.counts[k] += 1    # 更新选择次数
      self.actions.append(k)
      self.update_regret(k)  # 更新懊悔值
