# 🛡️ HSI Risk - 港股尾部風險評估系統

一個基於 **極端值理論 (EVT)** 與 **機器學習異常檢測** 的恆指風險評估工具。

## ✨ 功能
- 📥 自動下載 HSI 與 VHSI 數據
- 📉 計算 99% VaR 與 CVaR（歷史尾部風險）
- 📈 動態追蹤 Rolling VaR 走勢（歷史風險變化）
- 🕵️ 使用 Isolation Forest 檢測異常狀態（暴風雨前的寧靜）

## 🚀 快速開始
```bash
pip install -r requirements.txt
python main.py