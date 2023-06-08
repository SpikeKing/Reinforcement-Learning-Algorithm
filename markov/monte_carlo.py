#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/6/8
"""

import numpy as np


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


def sample(MDP, Pi, timestep_max, number):
    """
    采样函数
    :param MDP: MDP的元组
    :param Pi: 策略
    :param timestep_max: 最长时间步
    :param number: 采样的序列数
    :return: 全部采样
    """
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)   # 概率逐渐累加至1
                if temp > rand:  # 最终一定会选择某个动作 a_opt
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:  # 概率逐渐累加至1
                    s_next = s_opt  # 最终一定会跳转至下个状态s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes

# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


def main():
    np.random.seed(0)
    S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
    A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
    # 状态转移函数
    P = {
        "s1-保持s1-s1": 1.0,
        "s1-前往s2-s2": 1.0,
        "s2-前往s1-s1": 1.0,
        "s2-前往s3-s3": 1.0,
        "s3-前往s4-s4": 1.0,
        "s3-前往s5-s5": 1.0,
        "s4-前往s5-s5": 1.0,
        "s4-概率前往-s2": 0.2,
        "s4-概率前往-s3": 0.4,
        "s4-概率前往-s4": 0.4,
    }
    # 奖励函数
    R = {
        "s1-保持s1": -1,
        "s1-前往s2": 0,
        "s2-前往s1": -1,
        "s2-前往s3": -2,
        "s3-前往s4": -2,
        "s3-前往s5": 0,
        "s4-前往s5": 10,
        "s4-概率前往": 1,
    }
    gamma = 0.5  # 折扣因子
    MDP = (S, A, P, R, gamma)

    # 策略1,随机策略
    Pi_1 = {
        "s1-保持s1": 0.5,
        "s1-前往s2": 0.5,
        "s2-前往s1": 0.5,
        "s2-前往s3": 0.5,
        "s3-前往s4": 0.5,
        "s3-前往s5": 0.5,
        "s4-前往s5": 0.5,
        "s4-概率前往": 0.5,
    }

    # 采样5次,每个序列最长不超过20步
    episodes = sample(MDP, Pi_1, 20, 5)
    print('第一条序列\n', episodes[0])
    print('第二条序列\n', episodes[1])
    print('第五条序列\n', episodes[4])

    timestep_max = 20
    # 采样1000次,可以自行修改
    episodes = sample(MDP, Pi_1, timestep_max, 1000)
    gamma = 0.5
    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    MC(episodes, V, N, gamma)
    print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)

if __name__ == '__main__':
    main()