import pandas as pd


INPUT_FILE = 'stocks.xlsx'      # The name of your existing file
OUTPUT_FILE = 'filtered_stocks.xlsx' # The name of the new file to create
COLUMN_NAME = 'MARKET CAP'     # The exact name of your market cap column
THRESHOLD = 1


df = pd.read_excel("C:/Users/Shaad Hafeez/CODE/PYTHON/stocks.xlsx")
filtered = df[df[COLUMN_NAME] > THRESHOLD]

sorted_df = filtered.sort_values(by=COLUMN_NAME, ascending=False)
sorted_df.to_excel(OUTPUT_FILE, index=False)