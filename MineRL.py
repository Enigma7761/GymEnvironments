import gym
import minerl

if __name__ == "__main__":
    env = gym.make('MineRLNavigateDense-v0')
    model = TRPO(CnnLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("trpo_mine")

    del model # remove to demonstrate saving and loading

    model = TRPO.load("trpo_mine")

    obs = env.reset()

    done = False

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


