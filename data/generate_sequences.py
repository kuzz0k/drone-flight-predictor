import pandas as pd

def generate_windows(df: pd.DataFrame, window_size=6):
    X, y = [], []
    for start in range(len(df) - window_size + 1):
        block = df.iloc[start:start+window_size]
        X.append(block.iloc[:5].values)    # первые 5 точек
        y.append(block.iloc[5].values)     # 6-я точка
    return X, y
