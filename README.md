### SRA: Stochastic ranking-based multi-indicator algorithm

##### Reference: B. Li, K. Tang, J. Li, and X. Yao, Stochastic ranking algorithm for many-objective optimization based on multiple indicators, IEEE Transactions on Evolutionary Computation, 2016, 20(6): 924-938.

##### SRA is a many-objective evolutionary algorithm (MaOEA) which implements multiple indicators and stochastic ranking in environmental selection. This program implements SDE (shift-based density estimation) and $I_{\epsilon+}$.

| Variables   | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| npop        | Population size                                      |
| iter        | Iteration number                                     |
| lb          | Lower bound                                          |
| ub          | Upper bound                                          |
| T           | Neighborhood size (default = 30)                     |
| nobj        | The dimension of objective space (default = 3)       |
| eta_c       | Spread factor distribution index (default = 15)      |
| eta_m       | Perturbance factor distribution index (default = 15) |
| pt_min      | The minimum probability parameter (default = 0.4)    |
| pt_max      | The maximum probability parameter (default = 0.6)    |
| nvar        | The dimension of decision space                      |
| pop         | Population                                           |
| objs        | The objectives of population                         |
| mating_pool | Mating pool                                          |
| off         | Offspring                                            |
| off_objs    | The objective of offsprings                          |
| dom         | Domination matrix                                    |
| I1          | $I_{\epsilon+}$ indicator                            |
| I2          | SDE indicator                                        |

#### Test problem: LIR-CMOP6

$$
\begin{aligned}
& J_1=\{3, 5, \cdots, 29\}, J_2 = \{2, 4, \cdots, 30\} \\
& g_1(x) = \sum_{i \in J_1} (x_i - \sin(0.5i\pi x_1/30))^2 \\
& g_2(x) = \sum_{i \in J_2} (x_i - \cos(0.5i\pi x_1/30))^2 \\
&\min \\
& f_1(x) = x_1 + 10g_1(x) + 0.7057 \\
& f_2(x) = 1 - x_1^2 + 10g_2(x) + 0.7057 \\
& \text{subject to} \\
& c_k(x) = ((f_1 - p_k)\cos\theta - (f_2 - q_k)\sin\theta)^2/a_k^2 + ((f_1 - p_k)\sin\theta - (f_2 - q_k)\cos\theta)^2/b_k^2 \geq r \\
& p_k = [1.8, 2.8], \quad q_k = [1.8, 2.8], \quad a_k = [2, 2], \quad b_k = [8, 8], \quad k=1, 2 \\
& r = 0.1, \quad \theta = -0.25 \pi \\
& x_i \in [0, 1], \quad i = 1, \cdots, 30
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 300, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/SRA/blob/main/Pareto%20front.png)



