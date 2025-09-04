# RL-Trading (Mac Setup Guide)

這個專案是強化學習交易環境，支援 DQN / PPO / A2C。  
在 macOS (M1/M2/M3) 上可以使用 **MPS (Metal Performance Shaders)** 加速。  

---

## 1. 建立虛擬環境

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝 mac 專用 requirements
pip install -r requirements_mac.txt
