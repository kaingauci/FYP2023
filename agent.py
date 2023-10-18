import numpy as np
import random


class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0,
                 max_epsilon=1.0, min_epsilon=0.01, epsilon_decay_steps=1000, buffer_size=1000):
        # Number of possible actions the agent can take
        self.n_actions = n_actions

        # Learning rate for Q-learning updates
        self.lr = learning_rate

        # Discount factor for future rewards
        self.gamma = discount_factor

        # Initial exploration rate
        self.epsilon = exploration_rate

        # Maximum and minimum values for epsilon during decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        # Number of steps over which epsilon will decay
        self.epsilon_decay_steps = epsilon_decay_steps

        # Counter for the number of steps taken by the agent
        self.step_count = 0

        # Initialize Q-table with zeros for each state-action pair
        self.q_table = {
            "Decrease": [0, 0, 0],
            "Same": [0, 0, 0],
            "Increase": [0, 0, 0]
        }

        # Experience replay buffer to store experiences and learn from them
        self.buffer = []

        # Maximum size of the experience replay buffer
        self.buffer_size = buffer_size

    def choose_action(self, state):
        # With probability epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        # Otherwise, choose the action with the highest Q-value for the current state
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Store the recent experience in the replay buffer
        self.buffer.append((state, action, reward, next_state))

        # If the buffer exceeds its capacity, remove the oldest experience
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Randomly sample an experience from the buffer
        sample_state, sample_action, sample_reward, sample_next_state = random.choice(self.buffer)

        # Calculate the predicted Q-value from the Q-table
        predict = self.q_table[sample_state][sample_action]

        # Calculate the target Q-value using the sampled experience
        target = sample_reward + self.gamma * np.max(self.q_table[sample_next_state])

        # Update the Q-value in the Q-table using the Q-learning update rule
        self.q_table[sample_state][sample_action] += self.lr * (target - predict)

        # Linearly decay the exploration rate epsilon over time
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (
                    1 - self.step_count / self.epsilon_decay_steps)
        self.epsilon = max(self.min_epsilon, self.epsilon)

        # Optionally, linearly decay the learning rate over time
        self.lr *= 0.999
        self.lr = max(0.001, self.lr)  # Ensure the learning rate doesn't decrease too much

        # Increment the step counter
        self.step_count += 1
