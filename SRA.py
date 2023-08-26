#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 09:26
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : SRA.py
# @Statement : Stochastic ranking-based multi-indicator algorithm (SRA)
# @Reference : B. Li, K. Tang, J. Li, and X. Yao, Stochastic ranking algorithm for many-objective optimization based on multiple indicators, IEEE Transactions on Evolutionary Computation, 2016, 20(6): 924-938.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def environmental_selection(pop, objs, num, pt):
    # environmental selection
    npop = pop.shape[0]

    # calculate the indicator I1 (epsilon+)
    temp_I1 = np.zeros((npop, npop))
    for i in range(npop):
        for j in range(npop):
            temp_I1[i, j] = np.max(objs[i] - objs[j])
    I1 = np.sum(-np.exp(-temp_I1 / 0.05), axis=0)

    # calculate the indicator I2 (SDE)
    dis = np.full((npop, npop), np.inf)
    for i in range(npop):
        temp_objs = np.max((objs, np.tile(objs[i], (npop, 1))), axis=0)
        for j in range(i):
            dis[i, j] = np.sqrt(np.sum((objs[i] - temp_objs[j]) ** 2))
    I2 = np.min(dis, axis=1)

    # Stochastic ranking-based selection with bubble-sort
    rank = np.arange(npop)
    for _ in range(int(np.ceil(npop / 2))):
        flag = False
        for j in range(npop - 1):
            if np.random.random() < pt:  # I1
                if I1[rank[j]] < I1[rank[j + 1]]:
                    temp = rank[j]
                    rank[j] = rank[j + 1]
                    rank[j + 1] = temp
                    flag = True
            else:
                if I2[rank[j]] < I2[rank[j + 1]]:
                    temp = rank[j]
                    rank[j] = rank[j + 1]
                    rank[j + 1] = temp
                    flag = True
        if not flag:
            break
    return pop[rank[: num]], objs[rank[: num]]


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj=3, eta_c=15, eta_m=15, pt_min=0.4, pt_max=0.6):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 15)
    :param eta_m: perturbance factor distribution index (default = 15)
    :param pt_min: the minimum probability parameter (default = 0.4)
    :param pt_max: the maximum probability parameter (default = 0.6)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # the objectives of population

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = np.random.permutation(pop)
        off = crossover(mating_pool, lb, ub, 1, eta_c)
        off = mutation(off, lb, ub, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Stochastic ranking-based environmental selection
        pop, objs = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), npop, np.random.uniform(pt_min, pt_max))

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 300, np.array([0] * 7), np.array([1] * 7))
