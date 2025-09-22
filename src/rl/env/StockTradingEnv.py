import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

from rl.env.rewards import get_reward_fn

class StockTradingEnv:

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
        self.reward_fn = get_reward_fn(reward_mode)
        self.device = torch.device(device)
        self._timing_stats = None   # 每次 reset 初始化


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

        print(f"[DEBUG ENV INIT] obs_dim={self.obs_dim}, action_dim={self.action_dim}, features={df.columns.tolist()}")


    # endregion 初始化部分

    # region 小工具部分
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

    def _make_obs(self, t: int):
        start = time.time()

        feats = self.features[t - self.K + 1: t + 1]   # [K, N, F]
        feats = feats.permute(1, 2, 0)                 # [N, F, K]

        portfolio = self._weights_vector(t)            # [1+N]
        slot_info = self._slot_info(t)                 # [2*max_holdings]

        return {
            "features": feats,       # (N, F, K)
            "portfolio": torch.cat([portfolio, slot_info], dim=0)
        }

    # endregion 小工具部分

    # region GymAPI
    def reset(self, *, seed=None, options=None):
        stats_to_report = self._timing_stats  # 暫存上一輪的統計
        self._t = self.K
        self.cash = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.shares = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.avg_costs = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        #self.slots = [None] * self.max_holdings
        self.slots = torch.full((self.max_holdings,), -1, dtype=torch.long, device=self.device)
        self.portfolio_value = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.trade_count = 0

        obs = self._make_obs(self._t)                       # torch.Tensor
        action_mask_3d = self._build_action_mask(self._t)   # torch.BoolTensor

        info = {
            "V": self.portfolio_value.item(),   # 轉成 python int，方便 log
            "action_mask_3d": action_mask_3d    # 保留 torch tensor，PPO 可以直接用
        }

        # 初始化計時器
        self._timing_stats = {
            "check": 0.0,
            "prices": 0.0,
            "mask": 0.0,
            "trade": 0.0,
            "reward": 0.0,
            "obs": 0.0,
            "info": 0.0,
            "total": 0.0,
        }

        return obs, info

    def step(self, action):
        t_start = time.time()

        start = time.time()
        if self._t + 1 >= self.T:
            obs_now = self._make_obs(self._t)
            info_end = {"msg": "no next day", 
                        "V": int(self.portfolio_value),
                        "action_mask_3d": self._build_action_mask(self._t),
            }
            return obs_now, torch.tensor(0.0, device=self.device), True, False, info_end
        self._timing_stats["check"] += time.time() - start

        start = time.time()
        t = self._t

        p_open  = self.prices_open[t + 1]   # t+1 開盤成交
        p_close = self.prices_close[t + 1]  # t+1 收盤估值
        self._timing_stats["prices"] += time.time() - start

        op, idx, q = int(action[0]), int(action[1]), int(action[2])
        side, exec_shares, gross_cash, fees_tax = "HOLD", 0, 0, 0

        start = time.time()
        mask = self._build_action_mask(t)
        self._timing_stats["mask"] += time.time() - start

        # BUY
        start = time.time()
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
            self._timing_stats["trade"] += time.time() - start

        # Reward
        start = time.time()
        reward, reward_info = self.reward_fn(self, action, side, p_close, t)
        self._timing_stats["reward"] += time.time() - start

        
        # --- 生成觀測值 ---
        start = time.time()
        self._t += 1
        terminated = (self._t + 1 >= self.T)
        obs = self._make_obs(self._t)
        next_mask = self._build_action_mask(self._t)
        self._timing_stats["obs"] += time.time() - start
    
        # holdings_detail (輸出時轉回 python type)
        start = time.time()
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
            "trade_count": self.trade_count,
            "slots_mapping": {s: (self.ids[i] if i is not None else None) for s, i in enumerate(self.slots)},
            "holdings_detail": holdings_detail,
            **reward_info
        }
        self._timing_stats["info"] += time.time() - start

        self._timing_stats["total"] += time.time() - t_start

        """
        print("[PROFILE][Episode timing]")
        total = self._timing_stats["total"]
        for k, v in self._timing_stats.items():
            pct = (v / total * 100) if total > 0 else 0.0
            print(f"  {k:<8}: {v:.4f}s  ({pct:4.1f}%)")
        """

        return obs, reward, terminated, False, info
