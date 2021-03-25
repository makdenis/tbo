from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
from gym import wrappers, logger


class DQNAgent:
    def __init__(self, state_space, action_space, episodes=500):
        self.action_space = action_space
        self.memory_arr = []
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1.0 / float(episodes))
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n
        self.q_model = self.build(n_inputs, n_outputs)
        self.q_model.compile(loss="mse", optimizer=Adam())
        self.target_model = self.build(n_inputs, n_outputs)
        self.update_weights()
        self.replay_counter = 0

    def build(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs,), name="state")
        x = Dense(256, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(n_outputs, activation="linear", name="action")(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    def update_weights(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def to_act(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        q_values = self.q_model.predict(state)
        action = np.argmax(q_values[0])
        return action

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_arr.append(item)

    def get_target_value(self, next_state, reward):
        q_value = np.amax(self.target_model.predict(next_state)[0])
        q_value *= self.gamma
        q_value += reward
        return q_value

    def replay(self, batch_size):
        sars_batch = random.sample(self.memory_arr, batch_size)
        state_batch, q_values_batch = [], []
        for state, action, reward, next_state, done in sars_batch:
            q_values = self.q_model.predict(state)
            q_value = self.get_target_value(next_state, reward)
            q_values[0][action] = reward if done else q_value
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        self.q_model.fit(
            np.array(state_batch),
            np.array(q_values_batch),
            batch_size=batch_size,
            epochs=1,
            verbose=0,
        )

        self.update_epsilon()
        if self.replay_counter % 10 == 0:
            self.update_weights()
        self.replay_counter += 1

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    trials = 100
    win_reward = {"CartPole-v0": 195.0}
    scores = deque(maxlen=trials)
    logger.setLevel(logger.ERROR)
    env = gym.make("CartPole-v0")
    outdir = "/tmp/dqn-CartPole-v0"
    env = wrappers.Monitor(env, directory=outdir, video_callable=False, force=True)
    env.seed(0)
    agent = DQNAgent(env.observation_space, env.action_space)
    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 64
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        while not done:
            action = agent.to_act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if len(agent.memory_arr) >= batch_size:
            agent.replay(batch_size)

        scores.append(total_reward)
        mean_score = np.mean(scores)
        if mean_score >= win_reward["CartPole-v0"] and episode >= trials:
            print(
                f"Solved in episode {episode}: \
                   Mean survival ={mean_score} in {trials} episodes"
            )
            print("Epsilon: ", agent.epsilon)
            break
        if (episode + 1) % trials == 0:
            print(
                f"Episode {episode + 1}: Mean survival = \
                   {mean_score} in {trials} episodes"
            )
    env.close()
