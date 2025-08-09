
# backtest.py — quick CLI check that data loads
from data import load_data

if __name__ == "__main__":
    df = load_data("CL=F", "3mo", "1h")
    print(df.head())
    print("Rows:", len(df))
