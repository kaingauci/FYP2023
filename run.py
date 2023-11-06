import numpy as np
import matplotlib.pyplot as plt
from Data import load_data
from environment import StockTradingEnvironment
from agent import QLearningAgent

def main():
    # Load stock data for Apple and AMD from the given CSV files
    data_stock1 = load_data('AAPL.1HR.2Y.csv')
    data_stock2 = load_data('NVDA.1HR.2Y.csv')

    # Initialize the stock trading environment with the loaded data
    env = StockTradingEnvironment(data_stock1, data_stock2)

    # Initialize the Q-learning agent for both stocks with specified parameters
    agent_stock1 = QLearningAgent(n_actions=3, max_epsilon=1, min_epsilon=0.01, epsilon_decay_steps=1000, buffer_size=1000)
    agent_stock2 = QLearningAgent(n_actions=3, max_epsilon=1, min_epsilon=0.01, epsilon_decay_steps=1000, buffer_size=1000)

    # Lists to store metrics for analysis
    rewards = []  # Store total rewards for each episode
    percentage_in_stock1 = []  # Store percentage of money in stock 1 for each episode
    percentage_in_stock2 = []  # Store percentage of money in stock 2 for each episode
    cumulative_rewards = []  # Store cumulative rewards for each episode
    cumulative_reward = 100000  # Initialize cumulative reward to zero

    # Training loop for 100 episodes
    for episode in range(100):
        # Reset the environment and get the initial state for both stocks
        state_stock1, state_stock2 = env.reset()
        total_reward = 0

        # Episode loop
        while True:
            # Agent chooses an action based on the current state
            action_stock1 = agent_stock1.choose_action(state_stock1)
            action_stock2 = agent_stock2.choose_action(state_stock2)

            # Execute the chosen actions in the environment
            (next_state_stock1, reward_stock1, done1), (next_state_stock2, reward_stock2, done2) = env.step(action_stock1, action_stock2)

            # Agent learns from the results of the action
            agent_stock1.learn(state_stock1, action_stock1, reward_stock1, next_state_stock1)
            agent_stock2.learn(state_stock2, action_stock2, reward_stock2, next_state_stock2)

            # Update the current state for the next iteration
            state_stock1 = next_state_stock1
            state_stock2 = next_state_stock2

            # Update the total reward for this episode
            total_reward += reward_stock1 + reward_stock2

            # Exit the loop if the episode is done
            if done1 and done2:
                break

        # Update and store the cumulative reward for this episode
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)

        # Store metrics for analysis
        rewards.append(total_reward)
        total_money = env.cash_stock1 + env.cash_stock2 + (env.stock_owned_stock1 * env.last_price1) + (env.stock_owned_stock2 * env.last_price2)
        percentage_in_stock1.append((env.cash_stock1 + env.stock_owned_stock1 * env.last_price1) / total_money * 100)
        percentage_in_stock2.append((env.cash_stock2 + env.stock_owned_stock2 * env.last_price2) / total_money * 100)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # Plotting the results
    # Plot of total rewards and percentage of money in each stock over episodes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Trading Hours')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Over Episodes')

    plt.subplot(1, 2, 2)
    plt.plot(percentage_in_stock1, label='Apple')
    plt.plot(percentage_in_stock2, label='NVDA')
    plt.xlabel('Trading Hours')
    plt.ylabel('Percentage of Money (%)')
    plt.title('Percentage of Money in Each Stock Over Trading Hours')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot of episode rewards and cumulative rewards over episodes
    plt.plot(rewards, label="Episode Reward")
    plt.plot(cumulative_rewards, label="Portfolio value", linestyle='--')
    plt.xlabel('Trading Hours')
    plt.ylabel('Portfolio value')
    plt.title('Agent Performance Over Trading Hours')
    plt.legend()
    plt.show()

    # Print the final Q-table after training for both stocks
    print("\nFinal Q-Table for Stock 1 (Apple):")
    for state, actions in agent_stock1.q_table.items():
        print(f"State: {state}")
        for action, value in enumerate(actions):
            print(f"  Action {action}: {value}")

    print("\nFinal Q-Table for Stock 2 (NVDA):")
    for state, actions in agent_stock2.q_table.items():
        print(f"State: {state}")
        for action, value in enumerate(actions):
            print(f"  Action {action}: {value}")

if __name__ == "__main__":
    main()
