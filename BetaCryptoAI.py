import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QLabel, QPushButton, QLineEdit, QVBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import sys

def get_market_data(currency):
    # Replace YOUR_API_KEY with your own API key
    api_key = 'YOUR_API_KEY'
    
    # Request market data for the specified cryptocurrency from the API
    response = requests.get(f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={currency}&market=USD&apikey={api_key}')
    
    # Extract the data from the response
    data = response.json()['Time Series (Digital Currency Daily)']
    
    # Convert the data to a list of price points and additional features
    prices = []
    volumes = []
    market_caps = []
    for date in data:
        price = float(data[date]['4a. close (USD)'])
        volume = float(data[date]['5. volume'])
        market_cap = float(data[date]['6. market cap (USD)'])
        prices.append(price)
        volumes.append(volume)
        market_caps.append(market_cap)
    
    # Return the list of prices and additional features
    return prices, volumes, market_caps


def get_additional_features(prices, volumes, market_caps):
    # Calculate additional features such as moving averages and technical indicators
    features = []
    for i in range(len(prices)):
        # Calculate the moving average of the past 10 days
        if i > 9:
            moving_average = sum(prices[i-9:i+1]) / 10
            features.append(moving_average)
        
        # Calculate the relative strength index (RSI) of the past 14 days
        if i > 13:
            up_sum = 0
            down_sum = 0
            for j in range(i-13, i+1):
                if prices[j] > prices[j-1]:
                    up_sum += prices[j] - prices[j-1]
                else:
                    down_sum += prices[j-1] - prices[j]
            rsi = 100 - (100 / (1 + up_sum / down_sum))
            features.append(rsi)
    
    # Convert the list of features to a numpy array
    features = np.array(features)
    
    # Return the numpy array of features
    return features


def train_model(X, y, model_type, model_params=None):
    if model_type == 'SVM':
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
    elif model_type == 'Neural Network':
        model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if model_params:
            params = model_params.split(',')
            param_grid = {'hidden_layer_sizes': [(int(params[0]),)], 'alpha': [float(params[1])]}
            model = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True), n_jobs=-1)
    
    # Train the model on the data
    model.fit(X, y)
    
    # Return the trained model
    return model


def predict_price(model, X):
    # Make a prediction for the next price point
    predicted_price = model.predict(X.reshape(1, -1))[0]
    
    # Return the predicted price
    return predicted_price


class CryptocurrencyTrader(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Create the UI elements
        self.currency_label = QLabel('Currency:')
        self.currency_combo_box = QComboBox()
        self.currency_combo_box.addItems(['BTC', 'ETH', 'LTC'])
        self.model_label = QLabel('Model:')
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(['Linear Regression', 'Polynomial Regression', 'Random Forest'])
        self.params_label = QLabel('Model Parameters:')
        self.params_line_edit = QLineEdit()
        self.interval_label = QLabel('Interval (seconds):')
        self.interval_line_edit = QLineEdit()
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.status_label = QLabel('Ready')
        self.balance_label = QLabel('Balance: 100.00')
        self.trade_history_label = QLabel()
        self.plot_canvas = PlotCanvas()
        
        # Connect the button signals to the corresponding methods
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        
        # Create a layout for the UI elements
        grid = QGridLayout()
        grid.addWidget(self.currency_label, 0, 0)
        grid.addWidget(self.currency_combo_box, 0, 1)
        grid.addWidget(self.model_label, 1, 0)
        grid.addWidget(self.model_combo_box, 1, 1)
        grid.addWidget(self.params_label, 2, 0)
        grid.addWidget(self.params_line_edit, 2, 1)
        grid.addWidget(self.interval_label, 3, 0)
        grid.addWidget(self.interval_line_edit, 3, 1)
        grid.addWidget(self.start_button, 4, 0)
        grid.addWidget(self.stop_button, 4, 1)
        grid.addWidget(self.status_label, 5, 0, 1, 2)
        grid.addWidget(self.balance_label, 6, 0, 1, 2)
        grid.addWidget(self.trade_history_label, 7, 0, 1, 2)
        grid.addWidget(self.plot_canvas, 0, 2, 8, 1)
        
        # Set the layout for the main window
        self.setLayout(grid)
        
        # Set the window title and show the window
        self.setWindowTitle('Cryptocurrency Trader')
        self.show()

    def start(self):
        # Get the selected currency and model type
        currency = self.currency_combo_box.currentText()
        model_type = self.model_combo_box.currentText()
        model_params = self.params_line_edit.text()
        interval = self.interval_line_edit.text()
        
        if interval:
            interval = int(interval)
        else:
            interval = 60
        
        # Set the status label text
        self.status_label.setText(f'Updating market data for {currency}...')
        
        # Get the initial market data
        prices, volumes, market_caps = get_market_data(currency)
        X = get_additional_features(prices, volumes, market_caps)
        X = StandardScaler().fit_transform(X)
        y = prices
        
        # Train the model on the initial data
        model = train_model(X, y, model_type, model_params)
        
        # Set the initial balance and create an empty list to store the trade history
        balance = 100
        trade_history = []
        
        # Set the initial state to "holding"
        holding = False
        
        # Set the start time
        start_time = time.time()
        
        # Set the stop flag to False
        self.stop_flag = False
        
        # Enable the stop button
        self.stop_button.setEnabled(True)
        
        # Set the plot data to an empty list
        self.plot_data = []
        
        # Set the plot labels
        self.plot_canvas.set_labels('Time', 'Price')
    
        # Start the main loop
        while not self.stop_flag:
            # Update the market data
            prices, volumes, market_caps = get_market_data(currency)
            X = get_additional_features(prices, volumes, market_caps)
            X = StandardScaler().fit_transform(X)
            y = prices
        
        # Retrain the model on the updated data
        model = train_model(X, y, model_type, model_params)
        
        # Make a prediction for the next price point
        predicted_price = predict_price(model, X[-1, :])
        
        # Update the status label
        self.status_label.setText(f'Current price: {prices[-1]:.2f}, predicted price: {predicted_price:.2f}, balance: {balance:.2f}')
        
        # Update the plot data
        self.plot_data.append((time.time() - start_time, prices[-1]))
        self.plot_canvas.plot(self.plot_data)
        
        # Check if it's time to make a trade
        if not holding and predicted_price > prices[-1]:
            # Buy the cryptocurrency
            trade_history.append(('Buy', prices[-1]))
            balance -= prices[-1]
            holding = True
            self.balance_label.setText(f'Balance: {balance:.2f}')
            self.trade_history_label.setText('\n'.join([f'{trade[0]} at {trade[1]:.2f}' for trade in trade_history]))
        elif holding and predicted_price < prices[-1]:
            # Sell the cryptocurrency
            trade_history.append(('Sell', prices[-1]))
            balance += prices[-1]
            holding = False
            self.balance_label.setText(f'Balance: {balance:.2f}')
            self.trade_history_label.setText('\n'.join([f'{trade[0]} at {trade[1]:.2f}' for trade in trade_history]))
        
        # Wait for the specified interval
        time.sleep(interval)
    
    # Set the status label text
    self.status_label.setText('Stopped')
    
    # Disable the stop button
    self.stop_button.setEnabled(False)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


    def set_labels(self, xlabel, ylabel):
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)


    def plot(self, data):
        x = [point[0] for point in data]
        y = [point[1] for point in data]
        self.axes.clear()
        self.axes.plot(x, y)
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    trader = CryptocurrencyTrader()
    trader.show()
    sys.exit(app.exec_())
