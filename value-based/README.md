# Value Based Learning

## 1. Temporal Difference Learning

From **Discounted Return**

$$\begin{aligned}
G_t
& = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \cdots \\
& = R_t + \gamma \cdot (R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \cdots) \\
& = R_t + \gamma \cdot G_{t+1}
\end{aligned}$$

From **Action-value function**

$$\begin{aligned}
Q_{\pi}(s_t, a_t)
& = \mathbb{E}[G_t | s_t, a_t] \\
& = \mathbb{E}[R_t + \gamma \cdot G_{t+1} | s_t, a_t] \\
& = \mathbb{E}[R_t | s_t, a_t] + \gamma \cdot \mathbb{E}[G_{t+1} | s_t, a_t] \\
& = \mathbb{E}[R_t | s_t, a_t] + \gamma \cdot \mathbb{E}[Q_{\pi}(S_{t+1}, A_{t+1}) | s_t, a_t] \\
& = \mathbb{E}[R_t + \gamma \cdot Q_{\pi}(S_{t+1}, A_{t+1})] \\
\end{aligned}$$

Identity:

$$Q_{\pi}(s_t, a_t) = \mathbb{E}[R_t + \gamma \cdot Q_{\pi}(S_{t+1}, A_{t+1})], \text{ for all }\pi$$

And

1. We dont know $\mathbb{E}[R_t + \gamma \cdot Q_{\pi}(S_{t+1}, A_{t+1})]$
2. So we approximate it using Monte Carlo
3. use $r_t$ to approximate $R_t$
4. use $Q_{\pi}(s_{t+1}, a_{t+1})$ to approximate $Q_{\pi}(S_{t+1}, A_{t+1})$

So

$$\begin{aligned}
Q_{\pi}(s_t, a_t)
& = \mathbb{E}[R_t + \gamma \cdot Q_{\pi}(S_{t+1}, A_{t+1})] \\
& \approx r_t + \gamma \cdot Q_{\pi}(s_{t+1}, a_{t+1}) \\
\end{aligned}$$

And

1. We think $r_t + \gamma \cdot Q_{\pi}(s_{t+1}, a_{t+1})$ is more accurate because it contains the actual reward value $r_t$
2. Use [Temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning) to update $Q_{\pi}(s_t, a_t)$
3. So name $r_t + \gamma \cdot Q_{\pi}(s_{t+1}, a_{t+1})$ as **TD target** and $Q_{\pi}(s_t, a_t) - (r_t + \gamma \cdot Q_{\pi}(s_{t+1}, a_{t+1})$ as **TD error**
4. The purpose of **TD Learning** is encourage $Q_{\pi}(s_t, a_t)$ to approach **TD target**

## 2. Sarsa

use neural network $Q(s,a;\mathbf{w})$ to approximate **action-value function $Q_{\pi}(s,a)$** based on [Temporal Difference Learning](#1-temporal-difference-learning)

- **TD target**: $y_t = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1}; \mathbf{w})$
- **TD error**: $\delta_t = Q(s_t, a_t; \mathbf{w}) - y_t$
- **Loss**: $\frac{1}{2} \cdot {\delta_t}^2$

## 3. Temporal Difference Learning for Q-Learning

From [Temporal Difference Learning](#1-temporal-difference-learning):

$$Q_{\pi}(s_t, a_t) = \mathbb{E}[R_t + \gamma \cdot Q_{\pi}(S_{t+1}, A_{t+1})], \text{ for all }\pi$$

So

$$Q_{{\pi}^*}(s_t, a_t) = \mathbb{E}[R_t + \gamma \cdot Q_{{\pi}^*}(S_{t+1}, A_{t+1})]$$

> $Q_*(s_t,a_t)$ and $Q_{{\pi}^*}(s_t, a_t)$ are same, both denote **Optimal action-value function**

$$Q_*(s_t, a_t) = \mathbb{E}[R_t + \gamma \cdot Q_*(S_{t+1}, A_{t+1})]$$

As we always take best action

$$A_{t+1} = \underset{a}{\operatorname{argmax}}Q^*(S_{t+1}, a; \mathbf{w})$$

So

$$Q_*(S_{t+1}, A_{t+1}) = \max_a Q^* (S_{t+1}, a)$$

Thus

$$\begin{aligned}
Q_*(s_t, a_t)
& = \mathbb{E}[R_t + \gamma \cdot Q_*(S_{t+1}, A_{t+1})] \\
& = \mathbb{E}[R_t + \gamma \cdot \underset{a}{\operatorname{max}} Q^* (S_{t+1}, a)] \\
\end{aligned}$$

And

1. We dont know $\mathbb{E}[R_t + \gamma \cdot \underset{a}{\operatorname{max}} Q^* (S_{t+1}, a)]$
2. So we approximate it using Monte Carlo
3. use $r_t$ to approximate $R_t$
4. use $\underset{a}{\operatorname{max}} Q^*(s_{t+1}, a)$ to approximate $\underset{a}{\operatorname{max}} Q^*(S_{t+1}, a)$

So

$$\begin{aligned}
Q_*(s_t, a_t)
& = \mathbb{E}[R_t + \gamma \cdot \underset{a}{\operatorname{max}} Q^* (S_{t+1}, a)] \\
& \approx r_t + \gamma \cdot \underset{a}{\operatorname{max}} Q^*(s_{t+1}, a) \\
\end{aligned}$$

## 4. Q-Learning

use neural network $Q(s,a;\mathbf{w})$ to approximate **optimal action-value function $Q_*(s,a)$** based on [Temporal Difference Learning for Q-Learning](#3-temporal-difference-learning-for-q-learning)

- **TD target**: $y_t = r_t + \gamma \cdot \underset{a}{\operatorname{max}} Q(s_{t+1}, a)$
- **TD error**: $\delta_t = Q(s_t, a_t; \mathbf{w}) - y_t$
- **Loss**: $\frac{1}{2} \cdot {\delta_t}^2$
