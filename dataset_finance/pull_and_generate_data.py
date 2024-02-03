# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:04:46 2024

@author: bjaid
"""

import yfinance as yf
from textblob import TextBlob
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
company_dict = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "FB": "Meta Platforms, Inc.",  # formerly known as Facebook, Inc.
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)",
    "BRK.A": "Berkshire Hathaway Inc. (Class A)",
    "BRK.B": "Berkshire Hathaway Inc. (Class B)",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Company",
    "JPM": "JPMorgan Chase & Co.",
    "UNH": "UnitedHealth Group Incorporated",
    "MA": "Mastercard Incorporated",
    "INTC": "Intel Corporation",
    "VZ": "Verizon Communications Inc.",
    "HD": "Home Depot, Inc.",
    "DIS": "Walt Disney Company",
    "ADBE": "Adobe Inc.",
    "NFLX": "Netflix, Inc.",
    "PFE": "Pfizer Inc.",
    "BAC": "Bank of America Corporation",
    "CMCSA": "Comcast Corporation",
    "T": "AT&T Inc.",
    "MRK": "Merck & Co., Inc.",
    "PEP": "PepsiCo, Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "ABBV": "AbbVie Inc.",
    "NVDA": "NVIDIA Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "NKE": "NIKE, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "KO": "Coca-Cola Company",
    "DHR": "Danaher Corporation",
    "CVX": "Chevron Corporation",
    "LLY": "Eli Lilly and Company",
    "MDT": "Medtronic plc",
    "MCD": "McDonald's Corporation",
    "NEE": "NextEra Energy, Inc.",
    "ABT": "Abbott Laboratories",
    "COST": "Costco Wholesale Corporation",
    "QCOM": "Qualcomm Incorporated",
    "ACN": "Accenture plc",
    "TXN": "Texas Instruments Incorporated",
    "AVGO": "Broadcom Inc.",
    "LIN": "Linde plc",
    "UNP": "Union Pacific Corporation",
    "UPS": "United Parcel Service, Inc.",
    "LOW": "Lowe's Companies, Inc."
}

newsapi = NewsApiClient(api_key='21528994fb1c41388fb8dcdf53b4954f')
class fetchdata:
    def __init__(self):
        pass
    def fetch_newsapi_monthly(self, query, start_date, end_date, language='en', page_size=100):
        all_articles = newsapi.get_everything(q=query, language=language, sort_by='relevancy',page_size=page_size)
        sentiment_scores = []
        for article in all_articles['articles']:
            analysis = TextBlob(article['description'])
            sentiment_scores.append(analysis.sentiment.polarity)
        #return sentiment_scores
        #print(sum(sentiment_scores) / len(sentiment_scores))
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
fetchdataObj = fetchdata()
# Function to download stock data and generate additional variables
def fetch_and_generate(ticker, company_name):
    # Fetch historical data from Yahoo Finance (30 years range)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*365)
    data = yf.download(ticker, start="2001-01-01", end="2023-01-01")

    # Calculate day-to-day percentage change in closing prices
    data['Close_pct_change'] = data['Close'].pct_change() * 100

    # Calculate high-low price range
    data['High_Low_Range'] = data['High'] - data['Low']

    # Calculate 7-day moving average of closing prices
    data['7_day_MA'] = data['Close'].rolling(window=7).mean()

    # Calculate volume change
    data['Volume_Change'] = data['Volume'].diff()

    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Normalize Closing Price
    data['Normalized_Close'] = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())
    
    """
    sentiment_scores = pd.DataFrame(columns=['Date', 'Sentiment_Score'])
    
    # Iterate through each month
    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, 13):
            month_start = datetime(year, month, 1)
            month_end = datetime(year, month, 28)  # Simplified; adjust for each month's end date
    
            # Fetch sentiment score for the company for each month
            sentiment_score = fetchdataObj.fetch_newsapi_monthly(company_name + " COVID-19", month_start, month_end)
            index = len(sentiment_scores)
            # Append to the sentiment_scores DataFrame
            sentiment_scores.loc[index] = {'Date': month_end, 'Sentiment_Score': sentiment_score}
            #sentiment_scores = sentiment_scores.append({'Date': month_end, 'Sentiment_Score': sentiment_score}, ignore_index=True)
    
    # Convert 'Date' to datetime and set as index for merging
    sentiment_scores['Date'] = pd.to_datetime(sentiment_scores['Date'])
    sentiment_scores.set_index('Date', inplace=True)
    
    # Merge sentiment scores with stock data
    data = data.merge(sentiment_scores, how='left', left_index=True, right_index=True)

    return data
    """
    # Initialize DataFrame for sentiment scores
    sentiment_scores = pd.DataFrame(columns=['Date', 'Sentiment_Score'])
    
    covid_years = [2020, 2021, 2022]
    all_years = range(2001, 2024)
    
    # Calculate sentiment score only for COVID years monthly and yearly
    for year in all_years:
        if year in covid_years:
            for month in range(1, 13):
                try:
                    month_start = datetime(year, month, 1)
                    if month == 12:
                        month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                    else:
                        month_end = datetime(year, month + 1, 1) - timedelta(days=1)
    
                    sentiment_score = fetchdataObj.fetch_newsapi_monthly(company_name + " COVID-19", month_start, month_end)
                    index = len(sentiment_scores)
                    sentiment_scores.loc[index] = {'Date': month_end, 'Sentiment_Score': sentiment_score}
                    #sentiment_scores = sentiment_scores.append({'Date': month_end, 'Sentiment_Score': sentiment_score}, ignore_index=True)
                except Exception as e:
                    print(f"Error fetching sentiment score for {ticker} in {month}/{year}: {e}")
        else:
            # For non-COVID years, simply append None for each month
            for month in range(1, 13):
                month_end = datetime(year, month, 28)  # Simplified; adjust for each month's end date
                index = len(sentiment_scores)
                sentiment_scores.loc[index] = {'Date': month_end, 'Sentiment_Score': None}
                #sentiment_scores = sentiment_scores.append({'Date': month_end, 'Sentiment_Score': None}, ignore_index=True)
    
    # Convert 'Date' to datetime and set as index for merging
    sentiment_scores['Date'] = pd.to_datetime(sentiment_scores['Date'])
    data = data.merge(sentiment_scores.set_index('Date'), left_index=True, right_index=True, how='left')
    
    # Save the DataFrame to an Excel file
    #excel_filename = f"{ticker}_analysis.xlsx"
    #data.to_excel(excel_filename)
    #print(f"Saved data for {ticker} to {excel_filename}")
    
    return data
    
# List of 50 stock ticker symbols (you can modify this list as needed)
tickers = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "GOOG", "BRK.A", "BRK.B", "JNJ", "V", 
           "WMT", "PG", "JPM", "UNH", "MA", "INTC", "VZ", "HD", "DIS", "ADBE", 
           "NFLX", "PFE", "BAC", "CMCSA", "T", "MRK", "PEP", "TMO", "ABBV", "NVDA", 
           "CSCO", "NKE", "XOM", "KO", "DHR", "CVX", "LLY", "MDT", "MCD", "NEE", 
           "ABT", "COST", "QCOM", "ACN", "TXN", "AVGO", "LIN", "UNP", "UPS", "LOW"]

# Dictionary to store data for each ticker
stock_data = {}

# Fetch and generate data for each ticker
for ticker in company_dict:
    company_name = company_dict[ticker]
    print(f"Fetching data for {ticker}: {company_name}")
    stock_data[ticker] = fetch_and_generate(ticker, company_name)

# Example: Access data for Apple Inc.
print(stock_data["AAPL"].head())

excel_filename = "all_stock_data.xlsx"
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    for ticker, data in stock_data.items():
        data.to_excel(writer, sheet_name=ticker)

print(f"Saved all stock data to {excel_filename}")
