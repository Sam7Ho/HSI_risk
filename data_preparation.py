"""
data_preparation.py - 數據準備
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path


def fetch_data(start_date="2015-01-01", end_date=None):
    """獲取恆指與 VHSI 數據"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📥 獲取數據：{start_date} 到 {end_date}")
    
    # 下載數據
    hsi = yf.Ticker("^HSI").history(start=start_date, end=end_date, auto_adjust=False)
    vhsi = yf.Ticker("^HSIL").history(start=start_date, end=end_date, auto_adjust=False)
    
    hsi_df = hsi[['Close']].rename(columns={'Close': 'HSI_Close'})
    vhsi_df = vhsi[['Close']].rename(columns={'Close': 'VHSI_Close'})
    
    # ✅ 修正：用 outer + ffill
    df = pd.concat([hsi_df, vhsi_df], axis=1, join='outer')
    
    # 填充 VHSI 缺失值
    df['VHSI_Close'] = df['VHSI_Close'].ffill().bfill()
    
    # 移除 HSI 缺失的行（如果 HSI 都缺，這天就不是交易日）
    df = df.dropna(subset=['HSI_Close'])
    
    print(f"✅ 完成，共 {len(df)} 天")
    print(f"   HSI 範圍：{df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"   VHSI 缺失（已填充）：{vhsi_df['VHSI_Close'].isnull().sum()} 筆")
    
    return df


def save_data(df, path="data/raw_data.csv"):
    """儲存數據"""
    Path("data").mkdir(exist_ok=True)
    df.to_csv(path)
    print(f"💾 已儲存：{path}")


def load_data(path="data/raw_data.csv"):
    """載入數據"""
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ 找不到 {path}")
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"📂 載入數據：{path} ({len(df)} 筆)")
    return df


if __name__ == "__main__":
    df = fetch_data(start_date="2015-01-01")
    save_data(df)
    
    print(f"\n📊 數據集檢查：")
    print(f"   總樣本：{len(df)}")
    print(df.tail())