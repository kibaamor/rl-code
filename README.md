# Reinforcement Learning Code Snippet

[Understand the basic concepts in reinforcement learning](https://kibazen.cn/li-jie-qiang-hua-xue-xi-zhong-de-ji-ben-gai-nian/)

## HowToRun

1. install requirements

    ```bash
    pip install -r requirements.txt
    ```

2. run code

    ```bash
    cd value-based/tabular

    # test sarsa code
    python -m sarsa
    # if running in terminal without display, use blow command
    xvfb-run -a -s "-screen 0 1400x900x24" python -m raw_dqn
    ```

## 1. [Value Based Learning](https://kibazen.cn/qiang-hua-xue-xi-zhong-shi-xu-chai-fen-xue-xi/)

### 1.1 [Tabular Based](value-based/tabular/README.md)

1. [Sarsa](value-based/tabular/sarsa.py)
2. [Q-Learning](value-based/tabular/qlearning.py)

### 1.2 [Nerual Network Based](value-based/neural-network/README.md)

1. [Raw DQN](value-based/neural-network/raw_dqn.py)
2. [DQN with Experience Replay](value-based/neural-network/dqn_with_experience_replay.py)
3. Double DQN
4. Dualing DQN

## 2. Policy Based Learning

## 3. Actor-Critic
