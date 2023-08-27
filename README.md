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

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 300, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/SRA/blob/main/Pareto%20front.png)



