# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 02:25:34 2024

@author: bjaid
"""

import pandas as pd
import yfinance as yf
import os
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import re
import gzip
import shutil
import time

def get_empty_dataframe():
    return pd.DataFrame()

class fetch_dataV2:
    def __init__(self, pathToExcel, sheetName):
        self.excel_data = pd.read_excel(pathToExcel, sheet_name=sheetName)
        self.log_path = './output_files_v2/log.csv'  # Path to log file
        self.completed_tickers = self.read_log_file()  # Read completed tickers from log
    
    def read_log_file(self):
        """Reads the log file to determine which tickers have been completed."""
        if os.path.exists(self.log_path):
            log_df = pd.read_csv(self.log_path)
            return set(log_df['Ticker'].tolist())
        return set()

    def update_log_file(self, ticker, batch_file):
        """Updates the log file with the completed ticker and batch file."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(f"{ticker},{batch_file}\n")
        
    def fetch_OneCompany_data(self, ticker):
        # Fetch yearly data and retain date information
        # return yf.download(ticker, start="2001-01-01", end="2023-01-01", interval="1mo")
        data = yf.download(ticker, start="2001-01-01", end="2023-01-01", interval="1mo")
        data.reset_index(inplace=True)  # Reset the index to turn the date index into a column
        return data

    def fetch_yahoo_finance_data(self):
        esgDataFetcherObj = ESGDataFetcher()  # Initialize ESGDataFetcher
        batch_frames = []
        batch_count = self.get_next_batch_number()  # Determine the next batch number #1
        
        for index, row in self.excel_data.iterrows():
            ticker = row['Exchange:Ticker']
            if pd.notna(ticker) and ticker not in self.completed_tickers:
                print(f"*********Fetching data for: {ticker}*********")
                ticker_symbol = ticker.split(":")[1] if ":" in ticker else ticker
                esg_scores = esgDataFetcherObj.fetch_esg_scores(ticker_symbol)  # Fetch ESG scores
                
                data = self.fetch_OneCompany_data(ticker_symbol)
                # Replicate ESG scores for each year in data
                for col, score in esg_scores.items():
                    data[col] = score
                
                data['Company Name'] = row['Company Name']
                data['Industry Group'] = row['Industry Group']
                data['Primary Sector'] = row['Primary Sector']
                data['Country'] = row['Country']
                data['Ticker'] = ticker_symbol
                batch_frames.append(data)
                
                # After processing, update the log file
                self.update_log_file(ticker, f'out_batch_{batch_count}.csv')

                if len(batch_frames) == 100 or (index == len(self.excel_data) - 1):
                    batch_df = pd.concat(batch_frames, ignore_index=True)
                    batch_df.to_csv(f'./output_files_v2/out_batch_{batch_count}.csv', index=False)
                    batch_frames = []
                    batch_count += 1

    def get_next_batch_number(self):
        """Determines the next batch number to use based on existing files."""
        existing_batches = glob.glob('./output_files_v2/out_batch_*.csv')
        if existing_batches:
            latest_batch = max(existing_batches, key=os.path.getctime)
            latest_batch_number = int(re.search(r'out_batch_(\d+).csv', latest_batch).group(1))
            return latest_batch_number + 1
        return 1
    def combine_op_batch_files(self):
        pattern = os.path.join('./output_files_v2', 'out_batch_*.csv')
        csv_files = glob.glob(pattern)
        df_list = []
        for file in csv_files:
            df = pd.read_csv(file)
            df_list.append(df)
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_csv_path = os.path.join('./output_files_v2', 'output_us_yfinance.csv')
        combined_df.to_csv(combined_csv_path, index=False)
    def compress_final_file(self):
        # Specify the path of your CSV file and the output GZ file
        csv_file_path = 'output_files_v2/output_us_yfinance.csv'
        compressed_file_path = 'output_files_v2/output_us_yfinance.csv.gz'
        
        # Open the CSV file and the target GZ file
        with open(csv_file_path, 'rb') as input_file:
            with gzip.open(compressed_file_path, 'wb') as output_file:
                # Copy the contents of the CSV to the compressed file
                shutil.copyfileobj(input_file, output_file)
        
        print(f"Compressed file saved as: {compressed_file_path}")
    def decompress_final_file(self):
        # Specify the path of your compressed GZ file and the output CSV file
        compressed_file_path = 'output_files_v2/output_us_yfinance.csv.gz'
        decompressed_file_path = 'output_files_v2/output_us_yfinance.csv'
        
        # Open the compressed GZ file and the target CSV file
        with gzip.open(compressed_file_path, 'rb') as input_file:
            with open(decompressed_file_path, 'wb') as output_file:
                # Copy the contents of the compressed file to the decompressed file
                shutil.copyfileobj(input_file, output_file)
        
        print(f"Decompressed file saved as: {decompressed_file_path}")

class ESGDataFetcher:
    def __init__(self):
        options = webdriver.ChromeOptions()
        #options.add_argument('--headless')  # Run Chrome in headless mode
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    def fetch_esg_scores_not_working(self, ticker):
        self.driver.get(f"https://finance.yahoo.com/quote/{ticker}/sustainability")
        time.sleep(5)
        esg_scores = {'Environment Score': None, 'Social Score': None, 'Governance Score': None}
        try:
            # Update these XPaths with the correct ones for the actual data
            esg_scores['Environment Score'] = self.driver.find_element(By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[2]/div/div[2]').text
            esg_scores['Social Score'] = self.driver.find_element(By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[3]/div/div[2]/div[1]').text
            esg_scores['Governance Score'] = self.driver.find_element(By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[4]/div/div[2]/div[1]').text
        except Exception as e:
            print(f"Could not fetch ESG scores for {ticker} due to: {e}")
        return esg_scores
    

    def fetch_esg_scores(self, ticker):
        for i in range(10):
            try:
                self.driver.get("https://finance.yahoo.com/")
                break
            except Exception as e:
                pass
        try:
            # Wait for the search box to be clickable and enter the ticker
            search_box = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "yfin-usr-qry"))
            )
            search_box.clear()
            search_box.send_keys(ticker)
            search_box.send_keys(Keys.RETURN)
            #time.sleep(10)
            # Navigate to the 'Sustainability' tab
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Sustainability']"))
            ).click()

            # Wait for the ESG scores to be visible and extract them
            esg_scores = {
                'Environment Score': WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[2]/div/div[2]'))
                ).text,
                'Social Score': WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[3]/div/div[2]/div[1]'))
                ).text,
                'Governance Score': WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@id="Col1-0-Sustainability-Proxy"]/section/div[1]/div/div[4]/div/div[2]/div[1]'))
                ).text,
            }
            print(f"Fetched ESG scores for {ticker} {esg_scores}")
        except Exception as e:
            print(f"Could not fetch ESG scores for {ticker} as they don't exist")
            esg_scores = {'Environment Score': None, 'Social Score': None, 'Governance Score': None}

        return esg_scores
    
            
    def close_browser(self):
        self.driver.quit()

if __name__ == "__main__":
   fetchDataObj = fetch_dataV2('./data/indname.xls','US by industry')
   #fetchDataObj.fetch_yahoo_finance_data()
   #fetchDataObj.combine_op_batch_files()
   fetchDataObj.decompress_final_file()
