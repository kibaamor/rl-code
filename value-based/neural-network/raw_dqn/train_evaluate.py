from flappybird_wrapper import FlappyBirdWrapper
from raw_dqn import RawDQNAgent


def train_rawdqn(env, agent):
    total_step = 0
    total_reward = 0
    obs = env.reset()

    while True:
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        agent.learn(obs, act, reward, next_obs, done)

        obs = next_obs

        total_step += 1
        total_reward += reward

        if done:
            break

    return total_step, total_reward


def evaluate(env, agent):
    total_step = 0
    total_reward = 0
    obs = env.reset()

    while True:
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)

        total_step += 1
        total_reward += reward

        if done:
            break

    return total_step, total_reward


def train_evaluate(num):
    train_env = FlappyBirdWrapper()
    eval_env = FlappyBirdWrapper(display_screen=True)
    act_dim = train_env.action_space.n

    agent = RawDQNAgent(act_dim)
    for episode in range(num):
        total_step, total_reward = train_rawdqn(train_env, agent)
        print(
            f"train episode: {episode}, total step: {total_step}, total reward: {total_reward}"
        )

        if episode % 10 == 0:
            total_step, total_reward = evaluate(eval_env, agent)
            print(
                f"evaluate result. total step: {total_step}, total reward: {total_reward}"
            )


def main():
    num = 1000
    train_evaluate(num)


if __name__ == "__main__":
    main()
