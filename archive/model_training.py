"""
model_training.py - 模型訓練與驗證
功能：
1. 時間序列切分
2. 訓練隨機森林
3. 評估與儲存
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# 在你原本的 if __name__ == "__main__": 裡面
# 把 model = train_model(X_train, y_train) 換成：
# model = train_xgboost_model(X_train, y_train)

# 特徵清單
FEATURES = [
    'HSI_Return', 
    'Volatility_20', 
    'Volatility_60',
    'Dist_SMA_20', 
    'Dist_SMA_60',
    'Trend_Up',
    'VHSI_Change', 
    'VHSI_ZScore',
]

def train_xgboost_model(X_train, y_train):
    """訓練 XGBoost 模型"""
    print("\n🚀 訓練 XGBoost 模型...")
    
    # 計算負樣本與正樣本的比例，用來設定權重
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # 關鍵：讓模型重視黑天鵝
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print("✅ XGBoost 訓練完成")
    return model

def split_data(df, train_ratio=0.8):
    """時間序列切分"""
    split_idx = int(len(df) * train_ratio)
    
    X = df[FEATURES]
    y = df['Target']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📅 訓練集：{df.index[0].date()} ~ {df.index[split_idx-1].date()} ({len(X_train)} 筆)")
    print(f"📅 測試集：{df.index[split_idx].date()} ~ {df.index[-1].date()} ({len(X_test)} 筆)")
    print(f"📊 訓練集黑天鵝：{y_train.mean()*100:.2f}%")
    print(f"📊 測試集黑天鵝：{y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """訓練模型"""
    print("\n🤖 訓練隨機森林模型...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    print("✅ 訓練完成")
    return model


def evaluate_model(model, X_test, y_test):
    """評估模型"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("\n" + "="*50)
    print("📊 模型評估報告")
    print("="*50)
    print(f"AUC 分數：{roc_auc_score(y_test, y_prob):.3f}")
    print(f"精確率 (Precision)：{precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"召回率 (Recall)：{recall_score(y_test, y_pred, zero_division=0):.3f}")
    
    print("\n📋 詳細分類報告：")
    print(classification_report(y_test, y_pred, target_names=['正常', '黑天鵝']))
    
    print("\n📊 混淆矩陣：")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   真陰性 (TN)：{cm[0,0]}")
    print(f"   假陽性 (FP)：{cm[0,1]}")
    print(f"   假陰性 (FN)：{cm[1,0]}")
    print(f"   真陽性 (TP)：{cm[1,1]}")
    
    # 特徵重要性
    print("\n🔍 特徵重要性：")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"   {feat:<15}: {imp:.4f}")

    # 新增：Lift 分析
    print("\n📊 Lift 分析（前 10% 高風險預測）：")
    test_df = X_test.copy()
    test_df['Target'] = y_test
    test_df['Prob'] = y_prob

    # 取前 10% 最高機率的預測
    top_10_percent = test_df.nlargest(int(len(test_df) * 0.1), 'Prob')

    # 看這 10% 裡面有多少真的發生了黑天鵝
    hit_rate = top_10_percent['Target'].mean()
    baseline = y_test.mean()

    print(f"   前 10% 高風險預測中，黑天鵝發生率：{hit_rate*100:.2f}%")
    print(f"   測試集整體黑天鵝發生率：{baseline*100:.2f}%")
    print(f"   Lift（提升倍數）：{hit_rate / baseline:.2f} 倍")
    return {
        'auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'lift': hit_rate / baseline if baseline > 0 else 0,  # 避免除以 0
    }


def save_model(model, path="outputs/model.pkl"):
    """儲存模型"""
    Path("outputs").mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 模型已儲存：{path}")


def load_model(path="outputs/model.pkl"):
    """載入模型"""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_preparation import load_data
    from feature_engineering import create_features
    
    df = load_data()
    df = create_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_xgboost_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    
    print(f"\n✅ 完成！AUC：{metrics['auc']:.3f}")