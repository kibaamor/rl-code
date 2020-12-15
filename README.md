# Reinforcement Learning Code With Explanation

- Discounted Return
   $$G_t = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \cdots$$

- Action-value function
    $$Q_{\pi}(s_t,a_t) = \mathbb{E}[G_t | S_t=s_t,A_t=a_t]$$

- Optimal action-value function
    $$Q_*(s_t,a_t) = \max_{\pi} Q_{\pi}(s_t,a_t)$$

## 1. [Value Based Learning](value-based/README.md)

**Theory**: use neural network $Q(s,a;\mathbf{w})$ to approximate $Q_{\pi}(s,a)$ or $Q_*(s,a)$
**Use**: Always take best action $a = \underset{a}{\operatorname{argmax}}Q(s,a;\mathbf{w})$

### 1.1 Algorithm

1. Sarsa
    use neural network $Q(s,a;\mathbf{w})$ to approximate **action-value function $Q_{\pi}(s,a)$**.

2. Q-Learning
    use neural network $Q(s,a;\mathbf{w})$ to approximate **optimal action-value function $Q_*(s,a)$**.

### 1.2 Implemention

#### 1.2.1 Tabular Version

When the numbers of states and action are finite, we can use tables instead of neural networks.

1. [Sarsa](value-based/tabular/sarsa.py)
2. [Q-Learning](value-based/tabular/qlearning.py)

[Implemention Test](value-based/tabular/train_evaluate.py)

#### 1.2.2 Nerual Network Version

1. Deep Q-Learing(DQN)
2. Double DQN
3. Dualing DQN

## 2. Policy Based Learning

## 3. Actor-Critic
