class StockTradingEnvironment:
    def __init__(self, data_stock1, data_stock2, transaction_cost=0.0001):
        # Historical data for stock 1
        self.data1 = data_stock1
        # Historical data for stock 2
        self.data2 = data_stock2
        # Current step in the environment
        self.current_step = 0
        # Last known price for stock 1
        self.last_price1 = None
        # Last known price for stock 2
        self.last_price2 = None

        # Initial cash in hand for stock 1
        self.cash_stock1 = 100000 / 2  # Split initial cash between the two stocks
        # Initial cash in hand for stock 2
        self.cash_stock2 = 100000 / 2

        # Number of stock 1 shares owned
        self.stock_owned_stock1 = 0
        # Number of stock 2 shares owned
        self.stock_owned_stock2 = 0

        self.transaction_cost = transaction_cost

    def _discretize_state(self, current_open_price, previous_open_price):
        # Discretize the state based on the percentage change in open price
        if abs((current_open_price - previous_open_price) / previous_open_price) <= 0.003:
            return "Same"
        elif current_open_price < previous_open_price:
            return "Decrease"
        else:
            return "Increase"

    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        self.last_price1 = self.data1.iloc[self.current_step]['Open']
        self.last_price2 = self.data2.iloc[self.current_step]['Open']

        # Calculate the percentage change for both stocks
        change1 = (self.data1.iloc[self.current_step]['Open'] - self.data1.iloc[self.current_step - 1]['Open']) / \
                  self.data1.iloc[self.current_step - 1]['Open']
        change2 = (self.data2.iloc[self.current_step]['Open'] - self.data2.iloc[self.current_step - 1]['Open']) / \
                  self.data2.iloc[self.current_step - 1]['Open']

        # Calculate the total cash in hand
        total_cash = self.cash_stock1 + self.cash_stock2

        # Distribute the cash based on the ratio of the percentage changes
        self.cash_stock1 = total_cash * (change1 / (change1 + change2))
        self.cash_stock2 = total_cash * (change2 / (change1 + change2))

        # Reset the number of shares owned for both stocks
        self.stock_owned_stock1 = 0
        self.stock_owned_stock2 = 0

        return self._discretize_state(self.last_price1, self.last_price1), self._discretize_state(self.last_price2,
                                                                                                  self.last_price2)

    def step(self, action1, action2):
        # Take a step in the environment based on the actions provided for both stocks
        self.current_step += 1
        current_price1 = self.data1.iloc[self.current_step]['Open']
        current_price2 = self.data2.iloc[self.current_step]['Open']

        # Get the previous price for both stocks
        previous_price1 = self.data1.iloc[self.current_step - 1]['Open'] if self.current_step > 0 else self.last_price1
        previous_price2 = self.data2.iloc[self.current_step - 1]['Open'] if self.current_step > 0 else self.last_price2

        # Handle buy/sell actions for stock 1
        # Handle actions for stock 1
        if action1 == 0:
            stocks_to_buy = self.cash_stock1 // current_price1
            self.cash_stock1 -= stocks_to_buy * current_price1 * (
                        1 + self.transaction_cost)  # Deduct transaction cost for buying
            self.stock_owned_stock1 += stocks_to_buy
        elif action1 == 1:
            self.cash_stock1 += self.stock_owned_stock1 * current_price1 * (
                        1 - self.transaction_cost)  # Deduct transaction cost for selling
            self.stock_owned_stock1 = 0

            # Handle actions for stock 2
            if action2 == 0:
                stocks_to_buy = self.cash_stock2 // current_price2
                self.cash_stock2 -= stocks_to_buy * current_price2 * (
                            1 + self.transaction_cost)  # Deduct transaction cost for buying
                self.stock_owned_stock2 += stocks_to_buy
            elif action2 == 1:
                self.cash_stock2 += self.stock_owned_stock2 * current_price2 * (
                            1 - self.transaction_cost)  # Deduct transaction cost for selling
                self.stock_owned_stock2 = 0

        # Calculate the total portfolio value for both stocks
        next_portfolio_value1 = self.cash_stock1 + self.stock_owned_stock1 * current_price1
        next_portfolio_value2 = self.cash_stock2 + self.stock_owned_stock2 * current_price2

        # Calculate the reward based on the change in portfolio value
        reward1 = next_portfolio_value1 - (self.cash_stock1 + self.stock_owned_stock1 * self.last_price1)
        reward2 = next_portfolio_value2 - (self.cash_stock2 + self.stock_owned_stock2 * self.last_price2)

        # Update the last known prices for both stocks
        self.last_price1 = current_price1
        self.last_price2 = current_price2

        # Check if the episode is done (end of data or no money and stocks left)
        done = self.current_step == len(self.data1) - 1 or (self.cash_stock1 <= 0 and self.stock_owned_stock1 <= 0) or (
                    self.cash_stock2 <= 0 and self.stock_owned_stock2 <= 0)

        # Get the next state for both stocks
        next_state1 = self._discretize_state(current_price1, previous_price1)
        next_state2 = self._discretize_state(current_price2, previous_price2)

        return (next_state1, reward1, done), (next_state2, reward2, done)

