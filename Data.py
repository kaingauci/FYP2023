import pandas as pd

def load_data(filename):
    """Load the stock data from a CSV file."""
    data = pd.read_csv(filename)
    return data[['Open']]

def compute_rsi(data, window=14):
    """Compute the RSI (Relative Strength Index) for the given data."""
    # Calculate the difference between consecutive prices
    delta = data['Open'].diff()

    # Separate the gains and losses from the differences
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the average gain and average loss over the window period
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Example usage:
data = load_data('AAPL.1HR.2Y.csv')
data['RSI'] = compute_rsi(data)
print(data)
