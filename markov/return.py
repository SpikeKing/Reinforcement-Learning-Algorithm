#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/6/2
"""

import numpy as np

def compute_return(start_index, chain, gamma, rewards):
  """
  计算 Markov 链的回报
  """
  G = 0
  for i in reversed(range(start_index, len(chain))):
    G = gamma * G + rewards[chain[i] - 1]
  return G

def compute(P, rewards, gamma, states_num):
  """
  利用 贝尔曼方程 解析状态价值
  """
  rewards = np.array(rewards).reshape((-1, 1))  # 转换成列向量
  value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
  return value


def main():
  np.random.seed(0)

  P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  ]

  P = np.array(P)
  rewards = [-1, -2, -2, 10, 1, 0]
  gamma = 0.5


  chain = [1, 2, 3, 6]
  start_index = 0
  G = compute_return(start_index, chain, gamma, rewards)
  print(f"根据本次计算得到的回报是: {G}")

  V = compute(P, rewards, gamma, 6)
  print(f"MRP 中每个状态的价值分布是: {V}")

if __name__ == '__main__':
  main()
