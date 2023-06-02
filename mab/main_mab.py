#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2023/5/30
"""
from matplotlib import pyplot as plt

from decaying_epsilon_greedy import DecayingEpsilonGreedy
from epsilon_greedy import EpsilonGreedy
from mab.bernoulli_bandit import BernoulliBandit
from mab.thompson_sampling import ThompsonSampling
from mab.ucb import UCB


def plot_results(solvers, solver_names):
    """
    生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
        plt.xlabel('Time steps')
        plt.ylabel('Cumulative regrets')
        plt.title('%d-armed bandit' % solvers[0].bandit.K)
        plt.legend()
    plt.show()


def count_by_arr(max_prob, probs, counts):
    v_sum = 0.0
    for prob, count in zip(probs, counts):
        v_sum += (max_prob - prob) * count
    return v_sum

def single_eps():
    seed = 42
    K = 10
    bandit = BernoulliBandit(K, seed)
    epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print(f"eGreedy 算法的累积懊悔值: {epsilon_greedy_solver.regret}")
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    print(f"eGreedy 算法的最终期望值: {[round(i, 4) for i in epsilon_greedy_solver.estimates]}")
    print(f"eGreedy 算法的最终步数值: {list(epsilon_greedy_solver.counts)}")
    regret_2 = count_by_arr(bandit.best_prob, bandit.probs, epsilon_greedy_solver.counts)
    print(f"eGreedy 算法的累积懊悔值(计算): {regret_2}")


def multi_eps():
    seed = 42
    K = 10
    bandit = BernoulliBandit(K, seed)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit, epsilon=i) for i in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(i) for i in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
        plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    for i, solver in enumerate(epsilon_greedy_solver_list):
        print(f"{epsilons[i]} eGreedy 算法的最终步数值: {list(solver.counts)}")


def decay_eps():
    seed = 42
    K = 10
    bandit = BernoulliBandit(K, seed)

    epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
    epsilon_greedy_solver.run(5000)
    print(f"deGreedy 算法的累积懊悔值: {epsilon_greedy_solver.regret}")

    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    print(f"deGreedy 算法的最终步数值: {list(epsilon_greedy_solver.counts)}")


def ucb():
    seed = 42
    K = 10
    coef = 1
    bandit = BernoulliBandit(K, seed)
    ucb_solver = UCB(bandit, coef)
    ucb_solver.run(5000)
    print(f"UCB 算法的累积懊悔值: {ucb_solver.regret}")
    plot_results([ucb_solver], ["UCB"])


def thompson_sampling():
    seed = 42
    K = 10
    bandit = BernoulliBandit(K, seed)
    ts_solver = ThompsonSampling(bandit)
    ts_solver.run(5000)
    print(f"ThompsonSampling 算法的累积懊悔值: {ts_solver.regret}")
    plot_results([ts_solver], ["ThompsonSampling"])


def main_mab():
    # single_eps()
    # multi_eps()
    # decay_eps()
    # ucb()
    thompson_sampling()


if __name__ == '__main__':
  main_mab()
