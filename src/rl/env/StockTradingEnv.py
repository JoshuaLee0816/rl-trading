import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class StockTradingEnv:
    """
    StockTradingEnv (Broker-style, 1 trade per day)

    ▶ Observation (obs)
      shape = (N*F*K) + (1 + N) + (2 * max_holdings)
      = [K天 * N檔 * F特徵攤平] + [現金佔比] + [各股市值佔比(用t日收盤估值)] + [持倉槽位資訊]
      調用時序：在 t 觀測 → t 決策 → t+1 開盤成交 → t+1 收盤估值

    ▶ Action (MultiDiscrete([3, N, QMAX+1]))
      op  ∈ {0,1,2} = {BUY, SELL_ALL, HOLD}
      idx ∈ {0..N-1}   股票索引
      q   ∈ {0..QMAX}  買入張數（單位=張，=1000股；僅 BUY 時有效，q>=1 才有意義）
      * 每天只會執行一筆（買或全賣或不動） 
      * 不合法動作不懲罰，直接視為 HOLD（建議在 policy 端用 action_mask 做 masked softmax）

    ▶ Reward
      r_t = log( V_{t+1} / V_t )，以 t+1 收盤估值，已含交易成本/稅
      Set 0050 as baseline
      Penalty for high frequents trades -> cause log in r_t , so penalty would be set between 0.01 and 0.0001

    ▶ 限制
      - 只能整張交易（lot_size=1000）
      - 最多同時持有 max_holdings 檔（新開倉受限；既有持倉可加碼）
      - 交易成本：fee_buy / fee_sell；賣出另計 tax_sell

    ▶ reset() 回傳: (obs, info)
       info["action_mask"] 形狀 (3, N, QMAX+1)

    ▶ step(action) 回傳: (obs, reward, terminated, truncated, info)
       - terminated：到資料尾端
       - truncated：False（如需爆倉線可自行擴充）
       - info：紀錄當日動作、費稅、現金、持股數、下一步 action_mask
    """

    # region 初始化部分
    def __init__(
        self,
        df: pd.DataFrame,          # 長表: [date, stock_id, open, close, ...features]
        stock_ids,                 # 股票池順序（list-like）
        lookback: int,             # K 天視窗
        initial_cash: int,
        max_holdings: int,         # 同時持有檔數上限（新開倉受限；既有可加碼）
        qmax_per_trade: int = 10,  # 單日單筆最多可買的張數上限（固定維度用；實際受現金限制）
        seed: int = 42,
        fee_buy: float = 0.001425 * 0.6,
        fee_sell: float = 0.001425 * 0.6,
        tax_sell: float = 0.003,
        lot_size: int = 1000,       #一張 = 1000股
        reward_mode: str = "daily_return",
        action_mode: str = "discrete",
        device: str = "cpu"         # PyTorch 運算設備 (cpu/mps/cuda)
    ):

        # baseline 0050 (不能交易，只能計算報酬)
        df_baseline = df[df["stock_id"] == "0050"].copy()
        if df_baseline.empty:
            raise ValueError("df 缺少 baseline stock_id = 0050 ")
        df_baseline = df_baseline.sort_values("date")
        self.baseline_close = torch.tensor(
            df_baseline["close"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device
        )

        # 初始化參數
        self.ids = list(stock_ids)
        self.N = len(self.ids)
        self.K = int(lookback)
        self.initial_cash = float(initial_cash)
        self.max_holdings = int(max_holdings)
        self.QMAX = int(qmax_per_trade)
        self.fee_buy, self.fee_sell, self.tax_sell = float(fee_buy), float(fee_sell), float(tax_sell)
        self.lot_size = int(lot_size)
        self.rng = torch.Generator(device=device).manual_seed(seed)  # torch 替代 np.random
        self.reward_mode = reward_mode
        self.device = torch.device(device)

        # baseline 股票 (0050) 不能交易 → 提前建好 mask
        self.baseline_mask = torch.tensor(
            [sid == "0050" for sid in self.ids],
            dtype=torch.bool,
            device=self.device
        )

        # region 數據檢查
        df = df.copy().sort_values(["date", "stock_id"]).reset_index(drop=True)
        if not {"date", "stock_id", "open", "close"}.issubset(df.columns):
            raise ValueError("df 必須至少包含: ['date','stock_id','open','close']")
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

        missing = set(self.ids) - set(df["stock_id"].unique())
        if missing:
            raise ValueError(f"df 缺少以下 stock_ids: {sorted(missing)}")

        df = df[df["stock_id"].isin(self.ids)]
        self.dates = np.sort(df["date"].unique())
        self.T = len(self.dates)
        if self.T <= self.K + 1:
            raise ValueError("資料天數不足（需要 > lookback+1 天）")
        # endregion 數據檢查

        # In order to catch the features except date and stock_id 
        self._feat_cols = [c for c in df.columns if c not in ["date", "stock_id"]]

        # convert df's "open" and "close" into 2-dimensions tensors
        open_pv  = df.pivot(index="date", columns="stock_id", values="open").reindex(index=self.dates, columns=self.ids)
        close_pv = df.pivot(index="date", columns="stock_id", values="close").reindex(index=self.dates, columns=self.ids)
        self.prices_open  = torch.tensor(open_pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)   # [T, N]
        self.prices_close = torch.tensor(close_pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)  # [T, N]


        # 全特徵 3D -> torch tensor 
        feat_mats = []     #專門存放每個特徵轉換後的矩陣
        for c in self._feat_cols:
            pv = df.pivot(index="date", columns="stock_id", values=c).reindex(index=self.dates, columns=self.ids)
            arr = torch.tensor(pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)[..., None]  # [T, N, 1]
            feat_mats.append(arr)
        self.features = torch.cat(feat_mats, dim=2) if feat_mats else torch.zeros((self.T, self.N, 0), device=self.device)

        # 狀態 (用來描述當前狀態的變數)
        self._t = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self.avg_costs = None
        self.slots = None  # slot-based 持倉紀錄

        # === 不再定義 observation_space / action_space (移除 gym) ===
        self.obs_dim = self.N * self.features.shape[2] * self.K + (1 + self.N) + 2 * self.max_holdings
        self.action_dim = (3, self.N, self.QMAX + 1)  # MultiDiscrete 的結構 (直接用 tuple 儲存)

    # endregion 初始化部分

    # region 小工具部分
    """
    for loop 版本 耗時
    def _build_action_mask(self, t: int) -> torch.Tensor:
        
        #回傳形狀 (3, N, QMAX+1) 的布林遮罩 (torch.bool tensor)
        
        mask = torch.zeros((3, self.N, self.QMAX + 1), dtype=torch.bool, device=self.device)
        if t + 1 >= self.T:
            mask[2, 0, 0] = True
            return mask

        p_open = self.prices_open[t + 1]  # [N]
        baseline_mask = torch.tensor([sid == "0050" for sid in self.ids],
                                    dtype=torch.bool, device=self.device)

        # BUY
        for i in range(self.N):
            if baseline_mask[i]:
                continue
            price = p_open[i]
            if price <= 0:
                continue
            max_lots = int(self.cash // (self.lot_size * price * (1 + self.fee_buy)))
            max_q = min(max_lots, self.QMAX)
            if max_q >= 1:
                mask[0, i, 1:max_q + 1] = True

        # SELL_ALL
        for i in range(self.N):
            if baseline_mask[i]:
                continue
            if self.shares[i] > 0:
                mask[1, i, 0] = True

        # HOLD
        mask[2, 0, 0] = True
        return mask
    """

    # 向量化版本
    def _build_action_mask(self, t:int) -> torch.Tensor:
        #回傳形狀 (3, N, QMAX+1) 的布林遮罩 (torch.bool tensor)
        mask = torch.zeros((3, self.N, self.QMAX + 1), dtype=torch.bool, device=self.device)

        if t + 1 >= self.T:
            mask[2, 0, 0] = True
            return mask

        p_open = self.prices_open[t + 1]  # [N]

        # baseline 0050 mask（這個應該在 __init__ 就建好，這裡直接用 self.baseline_mask）
        baseline_mask = self.baseline_mask   # shape [N], True=不能交易

        # ============ BUY mask ============
        # 每檔股票可買的最大 lots 數
        max_lots = torch.floor(
            self.cash / (self.lot_size * p_open * (1 + self.fee_buy) + 1e-8)
        ).to(torch.int)

        # clamp 到 QMAX 範圍
        max_q = torch.clamp(max_lots, 0, self.QMAX)  # [N]

        # 建立一個 range [0..QMAX]
        q_range = torch.arange(self.QMAX + 1, device=self.device).view(1, -1)  # [1, QMAX+1]

        # 每檔股票允許的 q 值
        buy_mask = (q_range <= max_q.unsqueeze(1)) & (q_range >= 1)  # [N, QMAX+1]

        # baseline 股票強制 False
        buy_mask[baseline_mask] = False

        mask[0] = buy_mask

        # ============ SELL_ALL mask ============
        sell_mask = (self.shares > 0) & (~baseline_mask)  # [N]
        mask[1, sell_mask, 0] = True

        # ============ HOLD mask ============
        mask[2, 0, 0] = True
        return mask
    
    def _mark_to_market(self, prices: torch.Tensor) -> torch.Tensor:
        return (self.shares * prices).sum() + self.cash

    def _weights_vector(self, t: int) -> torch.Tensor:
        """
        輸出投資組合比例向量
        EX:
        現金    40%
        台積電  30%
        聯發科  20%
        鴻海    10%
        → 那麼 w = [0.4, 0.3, 0.2, 0.1]。
        """
        prices = self.prices_close[t]
        stock_val = (self.shares * prices).sum()
        V = stock_val + self.cash
        if V <= 0:
            w = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
            w[0] = 1.0
            return w
        return torch.cat([torch.tensor([self.cash / V], device=self.device),
                        (self.shares * prices) / V])

    def _features_window(self, t: int) -> torch.Tensor:
        win = self.features[t - self.K + 1: t + 1]   # [K, N, F]
        return win.reshape(-1)
    
    def _slot_info(self, t: int) -> torch.Tensor:
        """輸出 slot-based 部位資訊 [avg_cost, floating_return] * max_holdings"""
        hold_info = []
        for s in range(self.max_holdings):
            if self.slots[s] is None:
                hold_info.extend([0.0, 0.0])
            else:
                i = self.slots[s]
                avg_cost = self.avg_costs[i]
                price = self.prices_close[t, i]
                floating_ret = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
                hold_info.extend([avg_cost, floating_ret])
        return torch.tensor(hold_info, dtype=torch.float32, device=self.device)

    def _make_obs(self, t: int) -> torch.Tensor:
        feats = self._features_window(t)
        weights = self._weights_vector(t)
        slot_info = self._slot_info(t)
        obs = torch.cat([feats, weights, slot_info]).float()
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return torch.clamp(obs, -1e6, 1e6)
    # endregion 小工具部分

    # region GymAPI
    def reset(self, *, seed=None, options=None):
        self._t = self.K
        self.cash = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.shares = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.avg_costs = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.slots = [None] * self.max_holdings
        self.portfolio_value = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.trade_count = 0

        obs = self._make_obs(self._t)                       # torch.Tensor
        action_mask_3d = self._build_action_mask(self._t)   # torch.BoolTensor

        info = {
            "V": self.portfolio_value.item(),   # 轉成 python int，方便 log
            "action_mask_3d": action_mask_3d    # 保留 torch tensor，PPO 可以直接用
        }
        return obs, info

    def step(self, action):

        if self._t + 1 >= self.T:
            obs_now = self._make_obs(self._t)
            info_end = {"msg": "no next day", 
                        "V": int(self.portfolio_value),
                        "action_mask_3d": self._build_action_mask(self._t),
            }
            return obs_now, torch.tensor(0.0, device=self.device), True, False, info_end

        t = self._t

        p_open  = self.prices_open[t + 1]   # t+1 開盤成交
        p_close = self.prices_close[t + 1]  # t+1 收盤估值

        op, idx, q = int(action[0]), int(action[1]), int(action[2])
        side, exec_shares, gross_cash, fees_tax = "HOLD", 0, 0, 0
        mask = self._build_action_mask(t)

        # BUY
        if op == 0 and 0 <= idx < self.N and 1 <= q <= self.QMAX and mask[0, idx, q]:
            price = float(p_open[idx])

            # === 限制條件：低於 10 塊的股票不能買 ===
            if price < 10:
                side = "HOLD"   # 視為不動作
            
            else:
                lots  = min(int(q), int(self.cash // (self.lot_size * price * (1 + self.fee_buy))))
                if lots >= 1 and price > 0:
                    shares = lots * self.lot_size
                    gross  = int(round(shares * price))
                    fee    = int(round(gross * self.fee_buy))
                    cash_out = gross + fee
                    if cash_out <= self.cash + 1e-6:
                        old_shares = self.shares[idx]
                        self.shares[idx] += shares
                        self.cash -= cash_out
                        old_cost = self.avg_costs[idx]
                        if old_shares > 0:
                            self.avg_costs[idx] = (old_cost * old_shares + price * shares) / (old_shares + shares)
                        else:
                            self.avg_costs[idx] = price
                            for s in range(self.max_holdings):
                                if self.slots[s] is None:
                                    self.slots[s] = idx
                                    break
                        side, exec_shares, gross_cash, fees_tax = "BUY", shares, -gross, fee

        # SELL_ALL
        elif op == 1 and 0 <= idx < self.N and mask[1, idx, 0]:
            shares = int(self.shares[idx])
            if shares > 0 and p_open[idx] > 0:
                price = float(p_open[idx])
                gross = int(round(shares * price))
                fee   = int(round(gross * self.fee_sell))
                tax   = int(round(gross * self.tax_sell))
                cash_in = gross - fee - tax
                self.shares[idx] = 0
                self.cash += cash_in
                self.avg_costs[idx] = 0.0
                for s in range(self.max_holdings):
                    if self.slots[s] == idx:
                        self.slots[s] = None
                        break
                side, exec_shares, gross_cash, fees_tax = "SELL_ALL", -shares, gross, fee + tax
                self.trade_count += 1

        # Reward (思考是否需要reward scaling, 避免advantages過於快速接近0)
        V_prev = self.portfolio_value
        V_new  = self._mark_to_market(p_close)
        self.portfolio_value = V_new

        portfolio_return = torch.log(torch.clamp(V_new, min=1e-12) / torch.clamp(V_prev, min=1e-12))

        baseline_return = torch.log(
            self.baseline_close[t + 1] / self.baseline_close[t]
        )

        reward = portfolio_return - baseline_return
        if side in ("BUY", "SELL_ALL"):
            reward -= 0.00005

        # === 個股懲罰 (slot-based) ===
        penalty = torch.tensor(0.0, device=self.device)
        for s, i in enumerate(self.slots):
            if i is not None and self.avg_costs[i] > 0:
                cur_price = self.prices_close[self._t, i]
                floating_ret = (cur_price - self.avg_costs[i]) / self.avg_costs[i]
                if floating_ret < 0:
                    # 線性懲罰
                    # penalty += abs(floating_ret) * 0.05   # α=0.05，可調
                    # 指數型懲罰
                    penalty += (torch.exp(-5 * floating_ret) - 1)*0.001
        reward -= penalty
        
        # --- 生成觀測值 ---
        self._t += 1
        terminated = (self._t + 1 >= self.T)
        obs = self._make_obs(self._t)
        next_mask = self._build_action_mask(self._t)
    
        # holdings_detail (輸出時轉回 python type)
        holdings_detail = {
            s: {
                "stock_id": self.ids[i] if i is not None else None,
                "shares": int(self.shares[i].item()) if i is not None else 0,
                "avg_cost": float(self.avg_costs[i].item()) if i is not None else 0.0,
                "cur_price": float(self.prices_close[self._t, i].item()) if i is not None else 0.0,
                "floating_ret": (
                    float((self.prices_close[self._t, i] - self.avg_costs[i]) / self.avg_costs[i])
                    if (i is not None and self.avg_costs[i] > 0) else 0.0
                ),
            }
            for s, i in enumerate(self.slots)
        }

        info = {
            "V": int(self.portfolio_value.item()),
            "date": str(pd.Timestamp(self.dates[self._t]).date()),
            "side": side,
            "stock_id": (self.ids[idx] if 0 <= idx < self.N else None),
            "lots": (abs(exec_shares) // self.lot_size),
            "exec_shares": int(exec_shares),
            "gross_cash": int(gross_cash),
            "fees_tax": int(fees_tax),
            "cash": int(round(self.cash.item())),
            "held": int((self.shares > 0).sum().item()),
            "action_mask_3d": next_mask,
            "baseline_return": float(baseline_return.item()),
            "trade_count": self.trade_count,
            "slots_mapping": {s: (self.ids[i] if i is not None else None) for s, i in enumerate(self.slots)},
            "holdings_detail": holdings_detail,
        }

        return obs, reward, terminated, False, info
