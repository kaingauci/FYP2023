import numpy as np
import random
import matplotlib.pyplot as plt
from Data import load_data
from environment import StockTradingEnvironment
from agent import QLearningAgent

def main():
    # Load stock data for Apple from the given CSV file
    data = load_data('AAPL.1HR.2Y.csv')
    
    # Initialize the stock trading environment with the loaded data
    env = StockTradingEnvironment(data)
    
    # Initialize the Q-learning agent with specified parameters
    agent = QLearningAgent(n_actions=3, max_epsilon=1.0, min_epsilon=0.01, epsilon_decay_steps=1000, buffer_size=1000)

    # List to store total rewards for each episode
    rewards = []

    # Train the agent for 100 episodes
    for episode in range(100):
        # Reset the environment to its initial state at the beginning of each episode
        state = env.reset()
        total_reward = 0

        # Loop until the episode is done
        while True:
            # Agent chooses an action based on the current state
            action = agent.choose_action(state)
            
            # Execute the chosen action in the environment to get the next state, reward, and done flag
            next_state, reward, done = env.step(action)
            
            # Agent learns from the experience (state, action, reward, next_state)
            agent.learn(state, action, reward, next_state)
            
            # Update the current state to the next state
            state = next_state
            
            # Accumulate the reward for the episode
            total_reward += reward

            # If the episode is done, break out of the loop
            if done:
                break

        # Append the total reward for the episode to the rewards list
        rewards.append(total_reward)
        
        # Print the total reward for the episode
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # Plot the total rewards for all episodes to visualize the agent's performance over time
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Over Episodes')
    plt.show()

    # Print the final Q-table after training
    print("\nFinal Q-Table:")
    for state, actions in agent.q_table.items():
        print(f"State: {state}")
        for action, value in enumerate(actions):
            print(f"  Action {action}: {value}")

# Execute the main function if the script is run as the main module
if __name__ == "__main__":
    main()
