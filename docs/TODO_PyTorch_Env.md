# TODO: PyTorch_Env 移植計畫

## 🎯 目標
將 StockTradingEnv 與 PPOAgent 完整 PyTorch 化，最大化 MPS 加速效果。

---

## 🛠️ 待辦事項

- [ ] **環境改寫**
  - [ ] 將 `StockTradingEnv` 的 observation 從 numpy → torch.Tensor
  - [ ] action mask 改成 torch.bool tensor
  - [ ] reward / done flag 全部改成 torch scalar

- [ ] **Agent 修改**
  - [ ] `ppo_agent.py` 中 `select_action` 改用純 torch pipeline
  - [ ] rollout buffer 改成直接存 torch tensor（避免 numpy ↔ torch 轉換）
  - [ ] update loop 減少 CPU ↔ MPS 搬運

- [ ] **Train Loop**
  - [ ] `ppo_train.py` → `env.step()` 輸出直接維持在 torch
  - [ ] logger 改能處理 tensor / detach 到 cpu 再 log

- [ ] **測試**
  - [ ] 跑小型 dataset，確認 CPU / MPS 都能順利收斂
  - [ ] benchmark 執行速度 (CPU vs MPS)
  - [ ] 確認記憶體使用狀況是否改善

---

