# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:20:12 2024

@author: bjaid
"""
import pandas as pd
import yfinance as yf

def get_empty_dataframe():
    return pd.DataFrame()

class fetch_dataV2:
    def __init__(self, pathToExcel, sheetName):
        self.excel_data = pd.read_excel(pathToExcel, sheet_name=sheetName)
        
    def fetch_OneCompany_data(self, ticker):
        return yf.download(ticker, start="2001-01-01", end="2023-01-01", interval="1mo")
        
    def fetch_yahoo_finance_data(self):
        batch_frames = []  # Temporary storage for data frames within each batch
        batch_count = 1  # Counter to keep track of batches
        
        for index, row in self.excel_data.iterrows():
            ticker = row['Exchange:Ticker']
            # Check if the ticker is not NaN
            if pd.notna(ticker):
                print(f"*********Fetching data for: {ticker}*********")
                ticker_symbol = ticker.split(":")[1] if ":" in ticker else ticker
                data = self.fetch_OneCompany_data(ticker_symbol)
                data['Company Name'] = row['Company Name']
                data['Industry Group'] = row['Industry Group']
                data['Primary Sector'] = row['Primary Sector']
                data['Country'] = row['Country']
                data['Ticker'] = ticker_symbol
                batch_frames.append(data)
                
                # Check if we have processed 100 companies or if we are at the end of the dataframe
                if len(batch_frames) == 100 or (index == len(self.excel_data) - 1):
                    batch_df = pd.concat(batch_frames, ignore_index=True)
                    batch_df.to_csv(f'./output_files/out_batch_{batch_count}.csv', index=False)
                    batch_frames = []  # Reset the batch_frames for the next batch
                    batch_count += 1  # Increment the batch count


class fetch_data:
    def __init__(self, pathToExcel, sheetName):
        self.excel_data = pd.read_excel(pathToExcel, sheetName)
        self.ticker_symbols = self.excel_data['Exchange:Ticker'].unique()
        self.yahoo_finance_data = get_empty_dataframe()
        self.frames = []
    def fetch_OneCompany_data(self, ticker):
        return yf.download(ticker, start="2001-01-01", end="2023-01-01")
        #return yf.Ticker(ticker).history(period="20y", interval="1mo")
    def fetch_yahoo_finance_data(self):
        for index, row in self.excel_data.iterrows():
            ticker = row['Exchange:Ticker']
            # Check if the ticker is not NaN
            if pd.notna(ticker):
                print(f"*********Fetching data: {ticker}*********")
                ticker_symbol = ticker.split(":")[1] if ":" in ticker else ticker
                data = self.fetch_OneCompany_data(ticker_symbol)
                data['Company Name'] = row['Company Name']
                data['Industry Group'] = row['Industry Group']
                data['Primary Sector'] = row['Primary Sector']
                data['Country'] = row['Country']
                data['Ticker'] = ticker.split(":")[1]
                self.frames.append(data)
        self.yahoo_finance_data = pd.concat(self.frames, ignore_index=True)
        
            
if __name__ == "__main__":
   fetchDataObj = fetch_dataV2('./data/indname.xls','US by industry')
   #print(fetchDataObj.ticker_symbols)
   fetchDataObj.fetch_yahoo_finance_data()
   #fetchDataObj.yahoo_finance_data.to_csv('outv1.csv')
   