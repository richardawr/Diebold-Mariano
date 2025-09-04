MT5 Diebold-Mariano Test Analyzer

Overview
This Python script analyses MetaTrader 5 (MT5) Strategy Tester HTML reports and performs the Diebold-Mariano statistical test to compare the forecasting accuracy of a trading strategy against a benchmark.
Features

•	Parses MT5 HTML backtest reports to extract trade data
•	Calculates returns from the equity curve
•	Performs Diebold-Mariano test for predictive accuracy comparison
•	Supports multiple benchmark types:
o	CSV files with pre-calculated returns data
o	Another MT5 HTML report for EA-to-EA comparison
o	Raw MT5 price data CSV files (OHLC format)
o	Risk-free rate as default benchmark
•	Provides both command-line and graphical user interface (GUI) options

Requirements
Python Packages
•	pandas
•	numpy
•	beautifulsoup4
•	dieboldmariano (pip install dieboldmariano)
•	tkinter

Input Files
1.	MT5 HTML Report: Standard HTML output from MT5 Strategy Tester
2.	Benchmark Data (optional):
o	CSV file with pre-calculated returns data
o	Another MT5 HTML report for comparison
o	Raw MT5 price data CSV (OHLC format)


Usage
Command Line Interface

# Basic usage with default settings (risk-free rate benchmark)
python Diebold-Mariano.py path/to/mt5_report.html

# With CSV benchmark (pre-calculated returns)
python Diebold-Mariano.py path/to/mt5_report.html --benchmark-csv path/to/benchmark_returns.csv

# With another EA as benchmark
python Diebold-Mariano.py path/to/mt5_report.html --benchmark-html path/to/other_report.html

# With raw price data CSV (automatically calculates returns)
python Diebold-Mariano.py path/to/mt5_report.html --benchmark-csv path/to/price_data.csv --raw-prices

# Custom horizon
python Diebold-Mariano.py path/to/mt5_report.html --dm-horizon 2

# Force GUI mode
python Diebold-Mariano.py --gui
Graphical User Interface
1.	Run the script without arguments: python Diebold-Mariano.py
2.	Select your MT5 HTML report file
3.	Choose whether to use a benchmark:
o	Yes: Select either CSV file (returns or raw prices) or another HTML report
o	No: Uses risk-free rate (2% annualized) as benchmark
4.	Specify forecast horizon (default: 1)
5.	View results in console and pop-up window
Acceptable Input Formats
MT5 HTML Reports
•	Standard MT5 Strategy Tester HTML output
•	Must contain a "Deals" table with trade information
•	Should include columns: Time, Deal, Profit, Balance
•	Supports various encodings (UTF-16, UTF-8, CP1252, etc.)
Benchmark CSV Files - Two Types Supported:
1. Pre-calculated Returns
•	Should contain a column with return data
•	Acceptable column names: 'return', 'returns', 'ret', 'daily_return', 'daily_returns'
•	Alternatively, the first numeric column will be used
•	Returns should be in decimal format (e.g., 0.01 for 1%)
2. Raw MT5 Price Data (OHLC format)
•	CSV file exported directly from MT5
•	Should contain columns: Date, Time, Open, High, Low, Close, Volume
•	The script will automatically calculate daily returns from Close prices

Example format:
Date,Time,Open,High,Low,Close,Volume
2023.01.01,00:00,1.10000,1.10500,1.09900,1.10200,100000
2023.01.02,00:00,1.10200,1.10800,1.10100,1.10600,120000

Output Interpretation
The Diebold-Mariano test results include:
•	DM Statistic: Negative values favor the strategy, positive values favor the benchmark
•	p-value: Statistical significance (typically < 0.05 indicates significant difference)
•	Interpretation: Plain English explanation of the results

Result Interpretation Guide:
•	DM < 0 and p < 0.05: Strategy significantly outperforms benchmark
•	DM > 0 and p < 0.05: Benchmark significantly outperforms strategy
•	p ≥ 0.05: No significant difference between strategy and benchmark

Implementation Note for Raw Price Data
The script now includes functionality to handle raw MT5 price data CSV files. When a CSV file is provided with the --raw-prices flag (or detected as containing OHLC data), the script:
1.	Parses the date and time columns to create a proper datetime index
2.	Extracts closing prices
3.	Calculates daily percentage returns
4.	Aligns these returns with the strategy's trading timeline

This allows for direct comparison between your strategy's performance and the underlying asset's price movements.
Notes

•	The script handles various MT5 HTML formatting issues and encoding problems
•	Minimum of 2 trades required for returns calculation
•	For the DM test, at least horizon + 1 observations are needed
•	The script automatically trims benchmark data to match the strategy's length
•	When using raw price data, ensure the CSV covers the same time period as your backtest

Troubleshooting
1.	"dieboldmariano package not installed": Run pip install dieboldmariano
2.	"No valid deals found": Check if your HTML report contains trade data
3.	Encoding errors: The script tries multiple encodings automatically
4.	GUI not available: Use command-line arguments instead
5.	Price data alignment issues: Ensure your price CSV covers the same period as your backtest

Author
Richard, BLACK BOX LABS - 2025

Disclaimer
This tool is for educational and research purposes only. Past performance does not guarantee future results. Always validate trading strategies with multiple statistical tests and consider transaction costs, slippage, and other real-world factors.

