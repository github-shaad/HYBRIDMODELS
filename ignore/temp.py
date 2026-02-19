import yfinance as yf
import pandas as pd
from tqdm import tqdm
import warnings
import requests  # <-- Import the requests library
import json
# Suppress potential warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def fetch_stock_data(tickers):
    """
    Fetches detailed data for a list of tickers using yfinance.
    """
    all_stock_data = []
    
    print("Fetching financial data for each ticker...")
    for ticker in tqdm(tickers, desc="Fetching Data"):
        if not ticker:  # Skip empty strings from your file
            continue
            
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # --- THIS IS THE CORRECTED LINE ---
            # Bug 1: 'Exchange' -> 'exchange'
            # Bug 2: 'NYSE' -> 'NYQ'
            if (info.get('industry', 'N/A') == 'Telecom Services' and
                info.get('exchange', 'N/A') == 'NYQ' and
                info.get('sector', 'N/A') == 'Communication Services'):
                
                # Check for market cap before trying to convert to int
                market_cap_raw = info.get('marketCap')
                if market_cap_raw:
                    market_cap_billions = market_cap_raw / 1_000_000_000
                else:
                    market_cap_billions = 'N/A'

                data = {
                    'TICKER': ticker,
                    'NAME': info.get('longName', 'N/A'),
                    'MARKET CAP (Billions)': market_cap_billions,
                    'VOLUME': info.get('volume', 'N/A'),
                    'PE RATIO': info.get('trailingPE', 'N/A'),
                    'DIVIDEND YIELD': info.get('dividendYield', 'N/A'),
                    'BETA': info.get('beta', 'N/A'),
                }
                all_stock_data.append(data)

        except Exception as e:
            # This now properly handles tickers that 404 (like 'ANG')
            # You can comment out the print line if it's too noisy
            # print(f"Could not fetch/find data for {ticker}: {e}")
            pass # We just want to skip failed tickers, not record them
            
    return all_stock_data

def save_to_excel(data, filename):
    """
    Saves a list of dictionaries to an Excel file.
    """
    if not data:
        print("No data to save.")
        return

    print(f"\nConverting data to DataFrame...")
    df = pd.DataFrame(data)
    
    try:
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Successfully saved data to {filename}")
        print("\n--- Data Preview ---")
        print(df.head())
        print("--------------------")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

def get_tickers_json():
    try:
        with open("C:/Users/Shaad Hafeez/CODE/PYTHON/company_tickers.json", 'r') as file:
            loaded = json.load(file)    
            tickersList = [item["ticker"] for item in loaded.values()]
    except FileNotFoundError:
        print("File not found!")

    return tickersList    
def get_tickers():
    file = open("C:/Users/Shaad Hafeez/CODE/PYTHON/sp500_tickers.txt", "r")
    file = file.read()
    fileList = file.split("\n")
    return fileList

# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Define your filters
    SECTOR = "Communication Services"
    INDEX = "S&P 500"
    OUTPUT_FILE = "stocks3.xlsx"
    
    # 2. Get filtered list of tickers
    filtered_tickers = get_tickers()
    
    if filtered_tickers:
        # 3. Fetch data for those tickers
        stock_data_list = fetch_stock_data(filtered_tickers)
        
        # 4. Save the data to Excel
        save_to_excel(stock_data_list, OUTPUT_FILE)
    else:
        print("Script finished with no tickers to process.")
        