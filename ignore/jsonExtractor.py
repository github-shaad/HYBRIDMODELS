import json



def get_tickers_json():
    try:
        with open("C:/Users/Shaad Hafeez/CODE/PYTHON/company_tickers.json", 'r') as file:
            loaded = json.load(file)    
            
            tickersList = [item["ticker"] for item in loaded.values()]
            return tickersList
    except FileNotFoundError:
        print("File not found!")


