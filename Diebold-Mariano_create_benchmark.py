#!/usr/bin/env python3
"""
MT5 Diebold-Mariano Test Analyzer
Richard, BLACK BOX LABS, 2025
Parses MT5 Strategy Tester HTML reports and performs Diebold-Mariano test
"""

import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import sys

try:
    from dieboldmariano import dm_test

    DM_AVAILABLE = True
    # Check the function signature
    import inspect

    sig = inspect.signature(dm_test)
    print(f"dm_test signature: {sig}")
except ImportError:
    DM_AVAILABLE = False
    print("Warning: dieboldmariano package not installed. Install with: pip install dieboldmariano")


class MT5DMAnalyzer:
    def __init__(self, html_file_path):
        self.html_file_path = html_file_path
        self.deals_df = None
        self.returns = None
        self.benchmark_returns = None
        self.benchmark_start_date = None
        self.benchmark_end_date = None

    def parse_html_report(self):
        """Parse the MT5 HTML report and extract data"""
        # Try different encodings to handle various file formats
        encodings = ['utf-16', 'utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1', 'latin1']
        content = None

        for encoding in encodings:
            try:
                with open(self.html_file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding")

                # Check if the content looks corrupted (null bytes between characters)
                if content and any(ord(char) == 0 for char in content[:100]):
                    print(f"Content appears corrupted with {encoding} encoding, trying next...")
                    continue

                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("Could not decode the HTML file with any supported encoding")

        # If content still has null bytes, clean them up
        if content and any(ord(char) == 0 for char in content):
            print("Cleaning null bytes from content...")
            content = content.replace('\x00', '')

        try:
            soup = BeautifulSoup(content, 'html.parser')
            print("HTML parsing successful")

            # Extract deals data
            print("Extracting deals data...")
            self._extract_deals_data(soup)

            if self.deals_df is not None:
                print(f"✓ Parsed report: {len(self.deals_df)} deals found")
            else:
                print("✓ Parsed report: No deals dataframe created")

        except Exception as e:
            print(f"Error during parsing: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            raise

        return True

    def _parse_mt5_number(self, value_str):
        """Parse MT5 numbers that use spaces as thousand separators"""
        if not value_str or value_str == '-':
            return 0.0
        # Remove spaces (MT5 uses spaces as thousand separators) and commas
        clean_value = value_str.replace(' ', '').replace(',', '')
        try:
            return float(clean_value)
        except ValueError:
            return 0.0

    def _extract_deals_data(self, soup):
        """Extract deals data from HTML table"""
        print("Starting deals data extraction...")

        # Try different approaches to find tables
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables using find_all('table')")

        # Look for the Deals table specifically - use more specific criteria
        deals_table = None
        for i, table in enumerate(tables):
            # Get table text to check for specific Deals table content
            table_text = table.get_text()

            # Check if this table has the specific Deals header structure
            headers = table.find_all('th')
            header_text = ' '.join([h.get_text().strip() for h in headers])

            print(f"Table {i} headers: {header_text[:100]}...")

            # Look for the specific Deals table header pattern
            if ('Time' in header_text and 'Deal' in header_text and
                    'Profit' in header_text and 'Balance' in header_text):
                deals_table = table
                print(f"Found Deals table at index {i} by header pattern")
                break

            # Also check for the specific "Deals" section header
            section_headers = table.find_all('th', string=re.compile('Deals', re.IGNORECASE))
            if section_headers and len(section_headers) > 0:
                deals_table = table
                print(f"Found Deals table at index {i} by section header")
                break

        # If we didn't find it by headers, try to find by data pattern
        if not deals_table:
            print("Trying to find Deals table by data pattern...")
            for i, table in enumerate(tables):
                rows = table.find_all('tr')
                # Look for tables with many rows and specific data patterns
                if len(rows) > 10:  # Deals table should have many rows
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 10:  # Deals table should have many columns
                            first_cell = cells[0].get_text().strip()
                            # Look for datetime pattern which indicates deal data
                            if re.match(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}', first_cell):
                                deals_table = table
                                print(f"Found Deals table at index {i} by data pattern")
                                break
                    if deals_table:
                        break

        if not deals_table:
            # Print debug information about all tables
            print("Could not find Deals table. Available tables:")
            for i, table in enumerate(tables):
                print(f"=== Table {i} ===")
                headers = table.find_all('th')
                if headers:
                    header_text = ' | '.join([h.get_text().strip() for h in headers])
                    print(f"Headers: {header_text}")

                rows = table.find_all('tr')
                print(f"Rows: {len(rows)}")

                # Show first few rows
                for j, row in enumerate(rows[:3]):
                    cells = row.find_all(['td', 'th'])
                    cell_contents = [cell.get_text().strip()[:20] for cell in cells[:5]]  # First 5 cells
                    print(f"  Row {j}: {cell_contents}")

                print("---")

            # Try fallback to text extraction
            print("Falling back to text extraction...")
            text_content = soup.get_text()
            return self._extract_deals_from_text(text_content)

        # Now parse the deals data
        deals_data = []
        rows = deals_table.find_all('tr')
        data_started = False

        print(f"Deals table has {len(rows)} rows")

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text().strip() for cell in cells]

            # Look for the header row to start data parsing
            if not data_started:
                if ('Time' in cell_texts and 'Deal' in cell_texts and
                        'Profit' in cell_texts and 'Balance' in cell_texts):
                    data_started = True
                    print(f"Data starts at row {row_idx}")
                    continue
                else:
                    continue

            # Skip rows that don't have enough data or are summary rows
            if len(cell_texts) < 10:
                continue

            # Try to parse this row as a deal
            try:
                time_str = cell_texts[0]

                # Skip if it's not a valid datetime (might be summary row)
                if not re.match(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}', time_str):
                    continue

                # Parse the deal time
                deal_time = datetime.strptime(time_str, '%Y.%m.%d %H:%M:%S')

                # Extract deal data with proper error handling
                deal_data = {
                    'time': deal_time,
                    'deal_id': int(cell_texts[1]) if cell_texts[1].isdigit() else 0,
                    'symbol': cell_texts[2] if len(cell_texts) > 2 else '',
                    'type': cell_texts[3] if len(cell_texts) > 3 else '',
                    'direction': cell_texts[4] if len(cell_texts) > 4 else '',
                    'volume': self._parse_mt5_number(cell_texts[5]) if len(cell_texts) > 5 else 0.0,
                    'price': self._parse_mt5_number(cell_texts[6]) if len(cell_texts) > 6 else 0.0,
                    'order_id': int(cell_texts[7]) if len(cell_texts) > 7 and cell_texts[7].isdigit() else 0,
                    'commission': self._parse_mt5_number(cell_texts[8]) if len(cell_texts) > 8 else 0.0,
                    'swap': self._parse_mt5_number(cell_texts[9]) if len(cell_texts) > 9 else 0.0,
                    'profit': self._parse_mt5_number(cell_texts[10]) if len(cell_texts) > 10 else 0.0,
                    'balance': self._parse_mt5_number(cell_texts[11]) if len(cell_texts) > 11 else 0.0,
                    'comment': cell_texts[12] if len(cell_texts) > 12 else ''
                }

                # Skip balance entries if needed
                if deal_data['type'] != 'balance':
                    deals_data.append(deal_data)

            except (ValueError, IndexError) as e:
                continue

        print(f"Successfully parsed {len(deals_data)} deals")

        if len(deals_data) == 0:
            # Try one more approach with different column mapping
            print("Trying alternative column mapping...")
            return self._extract_deals_alternative_method(deals_table)

        self.deals_df = pd.DataFrame(deals_data)
        self.deals_df = self.deals_df.sort_values('time')
        self.deals_df.reset_index(drop=True, inplace=True)

        return True

    def _extract_deals_from_text(self, text_content):
        """Fallback method to extract deals from text content when tables can't be parsed"""
        print("Attempting to extract deals from text content...")

        # Look for lines that contain deal data
        lines = text_content.split('\n')
        deals_data = []

        for line in lines:
            # Look for lines with datetime pattern and numeric values
            if re.search(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}', line):
                # Try to extract values using various patterns
                parts = re.split(r'\s{2,}', line.strip())  # Split on multiple spaces
                if len(parts) >= 8:
                    try:
                        time_str = parts[0]
                        deal_time = datetime.strptime(time_str, '%Y.%m.%d %H:%M:%S')

                        deal_data = {
                            'time': deal_time,
                            'deal_id': int(parts[1]) if parts[1].isdigit() else 0,
                            'symbol': parts[2] if len(parts) > 2 else '',
                            'type': parts[3] if len(parts) > 3 else '',
                            'direction': parts[4] if len(parts) > 4 else '',
                            'volume': self._parse_mt5_number(parts[5]) if len(parts) > 5 else 0.0,
                            'price': self._parse_mt5_number(parts[6]) if len(parts) > 6 else 0.0,
                            'profit': self._parse_mt5_number(parts[10]) if len(parts) > 10 else 0.0,
                            'balance': self._parse_mt5_number(parts[11]) if len(parts) > 11 else 0.0,
                        }

                        deals_data.append(deal_data)
                    except (ValueError, IndexError):
                        continue

        print(f"Extracted {len(deals_data)} deals from text content")

        if len(deals_data) == 0:
            raise ValueError("No valid deals found in text content")

        self.deals_df = pd.DataFrame(deals_data)
        self.deals_df = self.deals_df.sort_values('time')
        self.deals_df.reset_index(drop=True, inplace=True)

        return True

    def _extract_deals_alternative_method(self, table):
        """Alternative method to extract deals with different column mapping"""
        print("Trying alternative extraction method...")

        deals_data = []
        rows = table.find_all('tr')
        data_started = False

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text().strip() for cell in cells]

            # Look for header row
            if not data_started:
                if any('Time' in text for text in cell_texts) and any('Deal' in text for text in cell_texts):
                    data_started = True
                continue

            if len(cell_texts) < 8:
                continue

            try:
                # Try different column patterns
                time_str = cell_texts[0]
                if not re.match(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}', time_str):
                    continue

                deal_time = datetime.strptime(time_str, '%Y.%m.%d %H:%M:%S')

                # Try to find profit and balance columns by pattern matching
                profit = 0.0
                balance = 0.0

                # Look for numeric values that could be profit/balance
                for i, text in enumerate(cell_texts):
                    if self._looks_like_amount(text):
                        if profit == 0.0:
                            profit = self._parse_mt5_number(text)
                        elif balance == 0.0:
                            balance = self._parse_mt5_number(text)

                deal_data = {
                    'time': deal_time,
                    'deal_id': int(cell_texts[1]) if cell_texts[1].isdigit() else row_idx,
                    'profit': profit,
                    'balance': balance
                }

                deals_data.append(deal_data)

            except (ValueError, IndexError) as e:
                continue

        print(f"Alternative method parsed {len(deals_data)} deals")

        if len(deals_data) == 0:
            raise ValueError("No valid deals found with any method")

        self.deals_df = pd.DataFrame(deals_data)
        self.deals_df = self.deals_df.sort_values('time')
        self.deals_df.reset_index(drop=True, inplace=True)

        return True

    def _looks_like_amount(self, text):
        """Check if text looks like a monetary amount"""
        if not text or text == '-':
            return False
        # Look for patterns like "100.00", "1 000.00", "-500.00"
        return bool(re.match(r'^[-]?[\d\s]+[.,]?\d*$', text.replace(' ', '')))

    def _is_returns_file(self, file_path):
        """Check if file is a returns CSV (has return/date columns)"""
        try:
            # Read first line to check columns
            with open(file_path, 'r') as f:
                first_line = f.readline().strip().lower()

            # Check for returns-related column names
            return_indicators = ['return', 'ret', 'daily_return', 'percent_change']
            return any(indicator in first_line for indicator in return_indicators)

        except:
            return False

    def get_trade_times(self):
        """Return array of datetime objects for each trade"""
        if self.deals_df is None:
            self.parse_html_report()

        # Filter out balance entries and non-trades, only include actual trading decisions
        trades = self.deals_df[
            (self.deals_df['type'] != 'balance') &
            (self.deals_df['type'] != 'commission') &
            (self.deals_df['type'] != 'swap')
            ]

        print(f"Found {len(trades)} actual trades for alignment")
        return trades['time'].values

    def load_benchmark_returns(self, benchmark_file_path):
        """
        Load benchmark returns from a CSV file

        Parameters:
        -----------
        benchmark_file_path : str
            Path to CSV file containing benchmark returns

        Returns:
        --------
        success : bool
            True if benchmark returns were loaded successfully
        """
        try:
            benchmark_df = pd.read_csv(benchmark_file_path)

            # Check for common column names
            return_col = None
            for col in ['return', 'returns', 'ret', 'daily_return', 'daily_returns']:
                if col in benchmark_df.columns:
                    return_col = col
                    break

            if return_col is None:
                # Try to find the first numeric column
                for col in benchmark_df.columns:
                    if pd.api.types.is_numeric_dtype(benchmark_df[col]):
                        return_col = col
                        break

            if return_col is None:
                print("Error: Could not find returns column in benchmark file")
                return False

            self.benchmark_returns = benchmark_df[return_col].values
            print(f"✓ Loaded {len(self.benchmark_returns)} benchmark returns from '{return_col}' column")
            return True

        except Exception as e:
            print(f"Error loading benchmark returns: {e}")
            return False

    def load_ea_benchmark(self, benchmark_html_path):
        """
        Load benchmark returns from another EA HTML report

        Parameters:
        -----------
        benchmark_html_path : str
            Path to MT5 HTML report to use as benchmark

        Returns:
        --------
        success : bool
            True if benchmark returns were loaded successfully
        """
        try:
            # Create a temporary analyzer to parse the benchmark EA
            benchmark_analyzer = MT5DMAnalyzer(benchmark_html_path)
            benchmark_analyzer.parse_html_report()
            benchmark_analyzer.calculate_returns()

            if benchmark_analyzer.returns is not None and len(benchmark_analyzer.returns) > 0:
                self.benchmark_returns = benchmark_analyzer.returns
                print(f"✓ Loaded {len(self.benchmark_returns)} benchmark returns from EA report")
                return True
            else:
                print("Error: Could not extract returns from benchmark EA report")
                return False

        except Exception as e:
            print(f"Error loading EA benchmark: {e}")
            return False

    def load_trade_aligned_benchmark(self, price_data_file):
        """
        Load benchmark returns aligned with EA trade times

        Parameters:
        -----------
        price_data_file : str
            Path to CSV file with price data (MT5 format)

        Returns:
        --------
        success : bool
            True if benchmark returns were loaded successfully
        """
        try:
            # Load price data
            print(f"Loading price data from: {price_data_file}")
            price_data = pd.read_csv(price_data_file, delim_whitespace=True)

            # Get trade times
            trade_times = self.get_trade_times()

            if len(trade_times) < 2:
                print("Not enough trades for alignment (need at least 2)")
                return False

            # Create benchmark returns aligned with trade times
            self.benchmark_returns = self._create_trade_aligned_returns(price_data, trade_times)
            print(f"✓ Created {len(self.benchmark_returns)} trade-aligned benchmark returns")
            return True

        except Exception as e:
            print(f"Error creating trade-aligned benchmark: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_trade_aligned_returns(self, price_data_df, trade_times):
        """
        Create benchmark returns between trade execution times
        """
        # Convert price data to datetime
        price_data_df['datetime'] = pd.to_datetime(
            price_data_df['<DATE>'] + ' ' + price_data_df['<TIME>'],
            format='%Y.%m.%d %H:%M:%S'
        )

        # Sort price data by datetime
        price_data_df = price_data_df.sort_values('datetime')

        benchmark_returns = []
        valid_trade_times = []

        print(f"Processing {len(trade_times)} trade times against {len(price_data_df)} price records")

        # For each trade interval (between trade i and trade i+1)
        for i in range(len(trade_times) - 1):
            start_time = trade_times[i]
            end_time = trade_times[i + 1]

            # Find price data between these trade times
            mask = (price_data_df['datetime'] >= start_time) & (price_data_df['datetime'] <= end_time)
            period_data = price_data_df[mask]

            if len(period_data) >= 2:
                # Calculate market return during this period
                start_price = period_data['<CLOSE>'].iloc[0]
                end_price = period_data['<CLOSE>'].iloc[-1]

                # Only include if we have valid prices
                if start_price > 0 and end_price > 0:
                    market_return = (end_price - start_price) / start_price
                    benchmark_returns.append(market_return)
                    valid_trade_times.append((start_time, end_time))
                else:
                    print(f"Warning: Invalid prices in period {start_time} to {end_time}")
            else:
                print(f"Warning: Insufficient data between {start_time} and {end_time}")

        print(f"Created {len(benchmark_returns)} trade-aligned benchmark returns")

        # Store date range for display
        if len(price_data_df) > 0:
            self.benchmark_start_date = price_data_df['<DATE>'].iloc[0]
            self.benchmark_end_date = price_data_df['<DATE>'].iloc[-1]

        return np.array(benchmark_returns)

    def debug_trade_alignment(self, price_data_file):
        """Debug method to identify the trade alignment issue"""
        print("\n=== DEBUG: TRADE ALIGNMENT ISSUE ===")

        try:
            # Load price data
            price_data = pd.read_csv(price_data_file, delim_whitespace=True)
            print(f"Price data records: {len(price_data)}")
            print(f"Price data columns: {list(price_data.columns)}")

            # Get trade times
            trade_times = self.get_trade_times()
            print(f"Trade times: {len(trade_times)}")

            # Check if we have the benchmark returns attribute
            if hasattr(self, 'benchmark_returns'):
                print(f"Benchmark returns length: {len(self.benchmark_returns)}")
            else:
                print("No benchmark returns created yet")

            # Check date formats
            print(f"First trade time: {trade_times[0]}")
            print(f"First price date: {price_data['<DATE>'].iloc[0]} {price_data['<TIME>'].iloc[0]}")

            # Check if trade times are within price data range
            price_data['datetime'] = pd.to_datetime(
                price_data['<DATE>'] + ' ' + price_data['<TIME>'],
                format='%Y.%m.%d %H:%M:%S'
            )

            print(f"Price data range: {price_data['datetime'].min()} to {price_data['datetime'].max()}")
            print(f"Trade times range: {trade_times.min()} to {trade_times.max()}")

            # Check for overlapping periods
            overlapping = np.sum((trade_times >= price_data['datetime'].min()) &
                                 (trade_times <= price_data['datetime'].max()))
            print(f"Trades within price data range: {overlapping}/{len(trade_times)}")

        except Exception as e:
            print(f"Debug error: {e}")
            import traceback
            traceback.print_exc()

    def calculate_returns(self):
        """Calculate returns series from equity curve"""
        if self.deals_df is None or len(self.deals_df) == 0:
            print("No deals data available for returns calculation")
            return None

        # Filter out balance entries and only use trade entries
        trade_df = self.deals_df[self.deals_df['type'] != 'balance']
        if len(trade_df) < 2:
            print("Not enough trade data for returns calculation")
            return None

        returns = []
        prev_balance = trade_df['balance'].iloc[0]

        for i in range(1, len(trade_df)):
            current_balance = trade_df['balance'].iloc[i]
            if prev_balance != 0:
                return_pct = (current_balance - prev_balance) / prev_balance
                returns.append(return_pct)
            prev_balance = current_balance

        self.returns = np.array(returns)
        print(f"✓ Calculated {len(self.returns)} returns from equity curve")
        return self.returns

    def calculate_diebold_mariano(self, horizon=1):
        """
        Calculate Diebold-Mariano test for predictive accuracy

        This test compares the forecasting accuracy of the strategy vs benchmark
        by treating them as competing forecasting methods.

        Parameters:
        -----------
        horizon : int, default=1
            Forecast horizon

        Returns:
        --------
        dm_result : dict
            Dictionary with DM test results
        """
        if not DM_AVAILABLE:
            print("Error: dieboldmariano package not available")
            return None

        if self.returns is None or len(self.returns) == 0:
            print("No returns data available for DM test")
            return None

        # Use risk-free rate as benchmark if no benchmark returns provided
        if self.benchmark_returns is None:
            # Create benchmark returns based on risk-free rate (default 2%)
            daily_rf = (1 + 0.02) ** (1 / 252) - 1
            benchmark_returns = np.full_like(self.returns, daily_rf)
            benchmark_name = "Risk-Free Rate (2%)"
        else:
            benchmark_returns = self.benchmark_returns
            benchmark_name = "Trade-Aligned Market"

            # Ensure lengths match
            min_length = min(len(self.returns), len(benchmark_returns))
            self.returns = self.returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]

            print(f"Using {min_length} matching periods for DM test")

        try:
            # For the DM test, we need actual values and two sets of predictions
            # We'll use a simple approach: treat the next period's return as the "actual"
            # and the current strategies' returns as "predictions"

            if len(self.returns) < horizon + 1:
                print(f"Not enough data for horizon {horizon}. Need at least {horizon + 1} observations.")
                return None

            # Create actual values (future returns)
            actual_values = self.returns[horizon:]

            # Create predictions (current strategy returns as forecasts)
            strategy_predictions = self.returns[:-horizon]
            benchmark_predictions = benchmark_returns[:-horizon]

            # Ensure all arrays have the same length
            min_len = min(len(actual_values), len(strategy_predictions), len(benchmark_predictions))
            actual_values = actual_values[:min_len]
            strategy_predictions = strategy_predictions[:min_len]
            benchmark_predictions = benchmark_predictions[:min_len]

            print(f"DM test using {min_len} observations with horizon {horizon}")

            # Calculate DM test with correct signature: dm_test(V, P1, P2, h=horizon)
            dm_result = dm_test(
                V=actual_values,
                P1=strategy_predictions,
                P2=benchmark_predictions,
                h=horizon
            )

            # Extract results - dm_test returns (dm_statistic, p_value)
            dm_stat, p_value = dm_result

            # Interpret results
            # DM test null hypothesis: both forecasts have equal accuracy
            # Positive DM stat means P1 (strategy) has higher loss than P2 (benchmark)
            # Negative DM stat means P1 (strategy) has lower loss than P2 (benchmark)

            if p_value < 0.05:
                if dm_stat < 0:
                    interpretation = f"Strategy significantly outperforms {benchmark_name} (p < 0.05)"
                else:
                    interpretation = f"{benchmark_name} significantly outperforms strategy (p < 0.05)"
            else:
                interpretation = f"No significant difference between strategy and {benchmark_name} (p ≥ 0.05)"

            dm_result_dict = {
                'dm_statistic': dm_stat,
                'p_value': p_value,
                'horizon': horizon,
                'interpretation': interpretation,
                'benchmark_name': benchmark_name,
                'observations_used': min_len
            }

            print(f"✓ Diebold-Mariano Test calculated: DM={dm_stat:.3f}, p={p_value:.3f}")
            return dm_result_dict

        except Exception as e:
            print(f"Error calculating Diebold-Mariano test: {e}")
            import traceback
            traceback.print_exc()
            return None

    def display_results(self, dm_result):
        """Display Diebold-Mariano test results with alignment info"""
        print("\n" + "=" * 70)
        print("           DIEBOLD-MARIANO TEST RESULTS")
        print("=" * 70)

        if self.deals_df is not None:
            trades = self.deals_df[
                (self.deals_df['type'] != 'balance') &
                (self.deals_df['type'] != 'commission') &
                (self.deals_df['type'] != 'swap')
                ]
            print(f"{'Strategy Trades:':<25} {len(trades):<20}")
            print(f"{'Strategy Returns:':<25} {len(self.returns) if self.returns is not None else 0:<20}")

        if self.benchmark_returns is not None:
            print(f"{'Benchmark Returns:':<25} {len(self.benchmark_returns):<20}")
            # Show date range if available
            if hasattr(self, 'benchmark_start_date') and hasattr(self, 'benchmark_end_date'):
                print(f"{'Benchmark Period:':<25} {self.benchmark_start_date} to {self.benchmark_end_date}")

        print("-" * 70)

        if dm_result:
            print(f"{'DM Statistic:':<25} {dm_result['dm_statistic']:<20.4f}")
            print(f"{'DM p-value:':<25} {dm_result['p_value']:<20.4f}")
            print(f"{'DM Horizon:':<25} {dm_result['horizon']:<20}")
            print(f"{'Observations Used:':<25} {dm_result['observations_used']:<20}")
            print(f"{'Benchmark:':<25} {dm_result['benchmark_name']:<20}")
            print()
            print(f"{'Interpretation:':<25}")
            print(f"  {dm_result['interpretation']}")
            print()
            print("Note: DM test compares forecasting accuracy.")
            print("Negative DM statistic favors the strategy.")
            print("Positive DM statistic favors the benchmark.")
        else:
            print("DM test could not be calculated")

        print("=" * 70)
        print(" " * 15 + "Richard, BLACK BOX LABS")
        print("=" * 70)


def select_html_file():
    """Open file dialog to select HTML file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select MT5 HTML Report",
        filetypes=[
            ("HTML files", "*.html"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path


def select_benchmark_file():
    """Open file dialog to select benchmark CSV file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select Benchmark CSV File",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path


def get_user_preferences():
    """Get user preferences via dialog boxes"""
    root = tk.Tk()
    root.withdraw()

    # Ask about benchmark type
    benchmark_type = messagebox.askyesno(
        "Benchmark Selection",
        "Would you like to use a benchmark for comparison?\n\n"
        "Yes: Select a benchmark file\n"
        "No: Use risk-free rate as benchmark"
    )

    benchmark_path = None
    ea_benchmark = False

    if benchmark_type:
        benchmark_choice = messagebox.askyesno(
            "Benchmark Type",
            "What type of benchmark would you like to use?\n\n"
            "Yes: CSV file with returns data\n"
            "No: Another EA HTML report"
        )

        if benchmark_choice:
            benchmark_path = select_benchmark_file()
        else:
            benchmark_path = select_html_file()
            ea_benchmark = True

    # DM test parameters
    dm_horizon = 1
    try:
        dm_horizon = simpledialog.askinteger(
            "DM Test Parameters",
            "Enter forecast horizon for DM test (default 1):",
            initialvalue=1,
            minvalue=1,
            maxvalue=10
        )
        if dm_horizon is None:
            dm_horizon = 1
    except:
        pass

    root.destroy()

    return {
        'benchmark_path': benchmark_path,
        'ea_benchmark': ea_benchmark,
        'dm_horizon': dm_horizon
    }


def main():
    """Main function with GUI file selection"""
    # Try command line arguments first, fall back to GUI
    parser = argparse.ArgumentParser(description='Perform Diebold-Mariano test on MT5 Strategy Tester HTML Report')
    parser.add_argument('html_file', nargs='?', default=None, help='Path to MT5 HTML report file (optional)')
    parser.add_argument('--benchmark-csv', help='CSV file with benchmark returns for DM test')
    parser.add_argument('--benchmark-html', help='HTML file with benchmark EA results for DM test')
    parser.add_argument('--price-data', help='CSV file with price data for trade-aligned benchmark')
    parser.add_argument('--dm-horizon', type=int, default=1, help='Forecast horizon for DM test')
    parser.add_argument('--gui', action='store_true', help='Force GUI mode')

    args = parser.parse_args()

    # Use GUI if no file specified or --gui flag used
    if not args.html_file or args.gui:
        print("Opening file selection dialog...")

        # Select HTML file
        html_file = select_html_file()
        if not html_file:
            print("No file selected. Exiting.")
            return 1

        # Get user preferences
        try:
            preferences = get_user_preferences()
        except tk.TclError:
            print("GUI not available. Using defaults.")
            preferences = {
                'benchmark_path': None,
                'ea_benchmark': False,
                'dm_horizon': 1
            }

        benchmark_path = preferences['benchmark_path']
        ea_benchmark = preferences['ea_benchmark']
        dm_horizon = preferences['dm_horizon']
        price_data_path = None

    else:
        # Use command line arguments
        html_file = args.html_file
        benchmark_path = args.benchmark_csv or args.benchmark_html
        ea_benchmark = bool(args.benchmark_html)  # True if benchmark is HTML, False if CSV or None
        dm_horizon = args.dm_horizon
        price_data_path = args.price_data

    # Check if file exists
    if not Path(html_file).exists():
        error_msg = f"Error: File '{html_file}' not found"
        print(error_msg)
        try:
            messagebox.showerror("File Error", error_msg)
        except:
            pass
        return 1

    # Create analyzer
    analyzer = MT5DMAnalyzer(html_file)

    try:
        # Parse HTML report
        print(f"Analyzing MT5 report: {Path(html_file).name}")
        analyzer.parse_html_report()

        # Calculate returns
        print("Calculating returns...")
        analyzer.calculate_returns()

        # Load benchmark data if provided
        if benchmark_path and os.path.exists(benchmark_path):
            print(f"Loading benchmark data from: {benchmark_path}")

            if ea_benchmark:
                analyzer.load_ea_benchmark(benchmark_path)
            else:
                # Auto-detect file type and load appropriately
                if analyzer._is_returns_file(benchmark_path):
                    print("Loading pre-calculated returns file...")
                    analyzer.load_benchmark_returns(benchmark_path)

                    # Trim benchmark to match strategy length if needed
                    if (analyzer.returns is not None and
                            analyzer.benchmark_returns is not None and
                            len(analyzer.benchmark_returns) > len(analyzer.returns)):
                        print(
                            f"Trimming benchmark returns ({len(analyzer.benchmark_returns)}) to match strategy ({len(analyzer.returns)})")
                        analyzer.benchmark_returns = analyzer.benchmark_returns[:len(analyzer.returns)]

                else:
                    print("Loading raw price data for trade-aligned benchmark...")
                    success = analyzer.load_trade_aligned_benchmark(benchmark_path)
                    if not success:
                        print("Trade-aligned benchmark failed, falling back to direct loading...")
                        analyzer.load_benchmark_returns(benchmark_path)

        # Calculate Diebold-Mariano test
        print("Performing Diebold-Mariano test...")
        dm_result = analyzer.calculate_diebold_mariano(horizon=dm_horizon)

        # Display results
        analyzer.display_results(dm_result)

        # Show completion message in GUI mode
        if not args.html_file or args.gui:
            try:
                message_text = "Diebold-Mariano test completed successfully!\n\n"
                if dm_result:
                    message_text += f"DM Statistic: {dm_result['dm_statistic']:.3f}\n"
                    message_text += f"DM p-value: {dm_result['p_value']:.3f}\n"
                    message_text += f"Interpretation: {dm_result['interpretation']}"

                messagebox.showinfo("Analysis Complete", message_text)
            except:
                pass

    except Exception as e:
        error_msg = f"Error analyzing report: {e}"
        print(error_msg)

        # Print full traceback for debugging
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

        try:
            messagebox.showerror("Analysis Error", error_msg)
        except:
            pass
        return 1

    return 0


if __name__ == "__main__":
    exit(main())