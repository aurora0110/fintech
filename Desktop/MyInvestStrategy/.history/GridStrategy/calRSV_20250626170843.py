import pandas as pd

def calrsv(
    df: pd.DataFrame,
    n: int,
) -> pd.Series:
    """
    衡量当前价格在特定周期内的相对位置
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["最低"].rolling(window=n, min_periods=1).min()
    high_close_n = df["收盘"].rolling(window=n, min_periods=1).max()
    rsv = (df["收盘"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    print(rsv.tolist()[-1])
    return rsv

if __name__ == "__main__":
    file_path = '/Users/lidongyang/Desktop/MYINVESTSTRATEGY/sh51030020250612.csv'
    data = pd.read_csv(file_path)

    calrsv(data, 9)
