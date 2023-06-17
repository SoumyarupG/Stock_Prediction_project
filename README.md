# Stock Trend Prediction using Keras and Streamlit

This project predicts stock trends using historical market prices and displays visualizations using Streamlit.

## Dependencies

- numpy
- pandas
- matplotlib
- pandas_datareader
- keras
- streamlit
- yfinance

## Description

The code in this repository is a stock trend prediction application that utilizes Keras and Streamlit. It allows users to enter a stock ticker and retrieve historical market prices for the specified stock. The code then performs data analysis, visualizes the closing price over time, and predicts future stock prices using a trained Keras model.

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file using pip or any package manager of your choice.
3. Download the Keras model file 'KerasModel.h5' and place it in the same directory as the code files.
4. Open a terminal or command prompt and navigate to the project directory.
5. Run the following command to start the application:
6. A web interface will open in your browser.
7. Enter the stock ticker you want to analyze in the provided text input field.
8. The application will retrieve the historical market prices and display descriptive statistics and visualizations.
9. It will also predict the stock prices for upcoming days and show the results in a separate chart.

## Notes

- The accuracy of the stock trend predictions may vary and should not be considered as financial advice.
- Make sure you have an active internet connection for the application to fetch the historical market prices using the Yahoo Finance API.

For any issues or questions, please contact [soumyarup.gh27@gmail.com]


