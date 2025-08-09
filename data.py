
# data.py — helper to fetch WTI (CL=F) from yfinance
import pandas as pd
import yfinance as yf

def load_data(symbol: str = "CL=F", period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
