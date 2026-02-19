import requests
from bs4 import BeautifulSoup
import sys

def fetch_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from the Wikipedia page.

    Returns:
        list: A list of ticker symbols (strings).
    """
    
    # The URL for the Wikipedia page
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Set a User-Agent header to mimic a web browser.
    # Wikipedia may block requests without a valid User-Agent.
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Fetching data from {WIKI_URL} ...")
    
    # Perform the HTTP GET request
    try:
        response = requests.get(WIKI_URL, headers=HEADERS)
        # Raise an exception for bad status codes (like 404 or 500)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}", file=sys.stderr)
        return []

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main table of constituents.
    # Based on inspection, the correct table has the ID "constituents".
    table = soup.find('table', {'id': 'constituents'})
    
    if table is None:
        print("Error: Could not find the constituents table on the page.", file=sys.stderr)
        print("The page structure might have changed.", file=sys.stderr)
        return []

    tickers = []
    
    # Find all table rows ('tr') within the table's body ('tbody')
    for row in table.find('tbody').find_all('tr'):
        # Find the first table cell ('td') in the row.
        # This cell contains the ticker symbol.
        # We skip the header row because it uses 'th' instead of 'td'.
        first_cell = row.find('td')
        
        if first_cell:
            # The ticker is the text of the first cell.
            # .strip() removes any leading/trailing whitespace or newlines.
            ticker = first_cell.text.strip()
            tickers.append(ticker)
            
    return tickers

if __name__ == "__main__":
    print("--- S&P 500 Ticker Scraper ---")
    
    try:
        ticker_list = fetch_sp500_tickers()
        
        if ticker_list:
            print(f"\nSuccessfully scraped {len(ticker_list)} tickers.")
            
            # --- Optional: Print the first 20 tickers ---
            print("First 20 tickers found:")
            print(", ".join(ticker_list[:20]))
            
            # --- Optional: Save all tickers to a file ---
            output_filename = "sp500_tickers.txt"
            with open(output_filename, 'w') as f:
                for ticker in ticker_list:
                    f.write(f"{ticker}\n")
            print(f"\nSuccessfully saved all tickers to {output_filename}")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)