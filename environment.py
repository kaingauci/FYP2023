# Defining a StockTradingEnvironment class to simulate a stock trading environment
class StockTradingEnvironment:
    # Initializing the environment with stock data
    def __init__(self, data):
        # Stock data (e.g., historical prices)
        self.data = data
        # Current timestep in the simulation
        self.current_step = 0
        # Last known stock price
        self.last_price = None
        # Initial cash in hand
        self.cash = 100000
        # Number of stocks owned
        self.stock_owned = 0

    def _discretize_state(self, current_open_price, previous_open_price):
        """Discretize the open price based on its relation to the previous day's open price."""
        # If the current price is less than the previous price, the price has decreased
        if current_open_price < previous_open_price:
            price_movement = "Decrease"
        # If the current price is greater than the previous price, the price has increased
        elif current_open_price > previous_open_price:
            price_movement = "Increase"
        # If the current price is the same as the previous price
        else:
            price_movement = "Same"
        return price_movement

    def reset(self):
        # Resetting the environment to its initial state
        self.current_step = 0
        # Setting the last price to the opening price of the first day
        self.last_price = self.data.iloc[self.current_step]['Open']
        # Resetting cash in hand
        self.cash = 100000
        # Resetting the number of stocks owned
        self.stock_owned = 0
        # For the first step, we'll use the same price as both current and previous
        return self._discretize_state(self.last_price, self.last_price)

    def step(self, action):
        # Move to the next timestep
        self.current_step += 1
        # Get the current opening price
        current_price = self.data.iloc[self.current_step]['Open']
        # Get the previous opening price
        previous_price = self.data.iloc[self.current_step - 1]['Open'] if self.current_step > 0 else self.last_price

        # If the action is to buy
        if action == 0:  # Buy
            # Calculate the number of stocks that can be bought with the current cash
            stocks_to_buy = self.cash // current_price
            # Deduct the cost of the stocks from the cash
            self.cash -= stocks_to_buy * current_price
            # Increase the number of stocks owned
            self.stock_owned += stocks_to_buy
        # If the action is to sell
        elif action == 1:  # Sell
            # Add the value of the stocks sold to the cash
            self.cash += self.stock_owned * current_price
            # Reset the number of stocks owned to zero
            self.stock_owned = 0
        # If action == 2, it's a "Hold" action, so no changes are made

        # Calculate the total value of the portfolio (cash + value of stocks)
        next_portfolio_value = self.cash + self.stock_owned * current_price
        # Calculate the reward as the difference in portfolio value from the last step
        reward = next_portfolio_value - (self.cash + self.stock_owned * self.last_price)
        # Update the last known price
        self.last_price = current_price

        # Check if the simulation has reached the end of the data
        done = self.current_step == len(self.data) - 1 or (
                    self.cash <= 0 and self.stock_owned <= 0)  # End episode on bankruptcy
        # Get the next state based on the current and previous prices
        next_state = self._discretize_state(current_price, previous_price)

        # Return the next state, reward, and whether the simulation is done
        return next_state, reward, done
