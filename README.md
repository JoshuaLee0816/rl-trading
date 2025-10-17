# RL-Trading (macOS PPO Research Project)

這個專案是針對台股資料設計的 **強化學習交易環境 (Reinforcement Learning Trading Environment)**。  
目前主要以 **PPO (Proximal Policy Optimization)** 為核心演算法，  
透過多種 Reward 模式與策略設定，研究在真實市場中的決策行為與穩定性。  

專案支援 **macOS (M1/M2/M3)**，可使用 **Metal Performance Shaders (MPS)** 進行 GPU 加速。  

---

## 1. 環境設定（macOS 專用）

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝 macOS 專用依賴
pip install -r requirements_mac.txt


## 2. 專案結構
RL_Trading/
│
├── src/
│   ├── rl/
│   │   ├── env/                # 強化學習環境（StockTradingEnv）
│   │   ├── models/             # PPO Actor / Critic 模型
│   │   ├── rewards/            # 自訂 Reward 函數（多種模式）
│   │   ├── train/              # 訓練主程式
│   │   └── test/               # 測試與績效分析
│   │
│   └── data/                   # 股票資料與 baseline
│
├── config.yaml                 # 訓練與環境參數設定
├── requirements_mac.txt        # macOS 相依套件
└── README.md
