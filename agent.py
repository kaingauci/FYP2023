import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.3, discount_factor=0.9, exploration_rate=0.7,
                 max_epsilon=0.9, min_epsilon=0.01, epsilon_decay_steps=1000, buffer_size=1000):
        # Number of possible actions
        self.n_actions = n_actions
        # Learning rate for Q-learning algorithm
        self.lr = learning_rate
        # Discount factor for future rewards
        self.gamma = discount_factor
        # Exploration rate for epsilon-greedy action selection
        self.epsilon = exploration_rate
        # Maximum exploration rate
        self.max_epsilon = max_epsilon
        # Minimum exploration rate
        self.min_epsilon = min_epsilon
        # Number of steps over which epsilon will decay
        self.epsilon_decay_steps = epsilon_decay_steps
        # Counter for the number of steps taken
        self.step_count = 0
        # Initialize Q-table with states and corresponding action values
        self.q_table = {
            "Decrease": [0, 0, 0],
            "Same": [0, 0, 0],
            "Increase": [0, 0, 0]
        }
        # Experience replay buffer to store past experiences
        self.buffer = []
        # Maximum size of the replay buffer
        self.buffer_size = buffer_size

    def choose_action(self, state):
        # Epsilon-greedy action selection
        # With probability epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        # Otherwise, choose the action with the highest Q-value for the current state
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Store the experience in the replay buffer
        self.buffer.append((state, action, reward, next_state))
        # If the buffer exceeds its capacity, remove the oldest experience
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        # Randomly sample an experience from the buffer
        sample_state, sample_action, sample_reward, sample_next_state = random.choice(self.buffer)
        # Predicted Q-value for the sampled state-action pair
        predict = self.q_table[sample_state][sample_action]
        # Target Q-value based on the reward and the maximum Q-value for the next state
        target = sample_reward + self.gamma * np.max(self.q_table[sample_next_state])
        # Update the Q-value using the Q-learning update rule
        self.q_table[sample_state][sample_action] += self.lr * (target - predict)
        # Decay the exploration rate over time
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (
                    1 - self.step_count / self.epsilon_decay_steps)
        # Ensure that the exploration rate does not go below the minimum
        self.epsilon = max(self.min_epsilon, self.epsilon)
        # Decay the learning rate over time
        self.lr *= 0.9
        # Ensure that the learning rate does not go below a threshold
        self.lr = max(0.001, self.lr)
        # Increment the step counter
        self.step_count += 1
