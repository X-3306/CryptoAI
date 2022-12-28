### THIS IS BETA VERSION


# BetaCryptoAI
this tool for predicting the price of a cryptocurrency using machine learning. It allows the user to select a cryptocurrency, a machine learning model, and 
some optional model parameters. The user can then start the tool, which will continuously update the market data for the selected cryptocurrency and use it to make 
predictions about the next price using the selected machine learning model. The tool also displays the current balance (profit or loss) based on the predicted prices.
The code first defines several functions for fetching and processing the market data for a cryptocurrency, training a machine learning model on the data, 
and making predictions using the trained model. It then defines a CryptocurrencyTrader class that represents the user interface for the tool. The user interface includes a 
combo box for selecting the cryptocurrency, a combo box for selecting the machine learning model, a line edit for entering optional model parameters, a line edit for entering 
the update interval, and start and stop buttons for controlling the tool.
The CryptocurrencyTrader class also includes a plot_canvas attribute, which is an instance of a PlotCanvas class that is responsible for displaying the history of trades made 
by the tool. The PlotCanvas class is a subclass of FigureCanvas from the matplotlib library, which is used to display plots in the user interface.
Finally, the code creates an instance of the CryptocurrencyTrader class and displays it to the user. It then enters the main event loop of the application, which waits for 
user input and responds to events such as button clicks.

# how to run?
pip install -r requirements.txt
python3 CryptoAI.py

