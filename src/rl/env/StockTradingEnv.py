import json
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces

from rl.env.rewards import get_reward_fn


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_ids,
        lookback: int,
        initial_cash: int,
        max_holdings: int,
        qmax_per_trade: int = 10,
        seed: int = 42,
        fee_buy: float = 0.001425 * 0.6,
        fee_sell: float = 0.001425 * 0.6,
        tax_sell: float = 0.003,
        lot_size: int = 1000,
        reward_mode: str = "daily_return",
        action_mode: str = "discrete",
        device: str = "cpu",        # ⚠️ 環境在CPU，Agent在MPS/GPU
    ):
        # baseline 0050 必須存在
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
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.reward_fn = get_reward_fn(reward_mode)
        self.device = torch.device(device)
        self._timing_stats = None

        self.baseline_mask = torch.tensor(
            [sid == "0050" for sid in self.ids],
            dtype=torch.bool,
            device=self.device
        )

        # 數據檢查與重排
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

        # 特徵矩陣
        self._feat_cols = [c for c in df.columns if c not in ["date", "stock_id"]]
        open_pv  = df.pivot(index="date", columns="stock_id", values="open").reindex(index=self.dates, columns=self.ids)
        close_pv = df.pivot(index="date", columns="stock_id", values="close").reindex(index=self.dates, columns=self.ids)
        self.prices_open  = torch.tensor(open_pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)
        self.prices_close = torch.tensor(close_pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)

        feat_mats = []
        for c in self._feat_cols:
            pv = df.pivot(index="date", columns="stock_id", values=c).reindex(index=self.dates, columns=self.ids)
            arr = torch.tensor(pv.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)[..., None]
            feat_mats.append(arr)
        self.features = torch.cat(feat_mats, dim=2) if feat_mats else torch.zeros((self.T, self.N, 0), device=self.device)

        # 狀態
        self._t = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self.avg_costs = None
        self.slots = None  # list[int|None]

        # 觀測/動作空間（必要：供 AsyncVectorEnv 檢查）
        self.obs_dim = self.N * self.features.shape[2] * self.K + (1 + self.N) + 2 * self.max_holdings
        self.action_dim = (3, self.N, self.QMAX + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(self.action_dim)

        # 這個不要刪 很有用
        #print(f"[DEBUG ENV INIT] obs_dim={self.obs_dim}, action_dim={self.action_dim}, features={df.columns.tolist()}")

    # ---- 工具 ----
    def _build_action_mask(self, t:int) -> torch.Tensor:
        mask = torch.zeros((3, self.N, self.QMAX + 1), dtype=torch.bool, device=self.device)
        if t + 1 >= self.T:
            mask[2, 0, 0] = True
            return mask

        p_open = self.prices_open[t + 1]
        baseline_mask = self.baseline_mask

        max_lots = torch.floor(self.cash / (self.lot_size * p_open * (1 + self.fee_buy) + 1e-8)).to(torch.int)
        max_q = torch.clamp(max_lots, 0, self.QMAX)
        q_range = torch.arange(self.QMAX + 1, device=self.device).view(1, -1)

        # ✅ 限制開新倉：若已達 max_holdings，只允許對現有部位加碼
        held_mask = (self.shares > 0) & (~baseline_mask)
        held_count = int(held_mask.sum().item())
        can_open_new = (held_count < self.max_holdings)

        allow_idx = held_mask | (torch.tensor(can_open_new, device=self.device) & (~baseline_mask))

        buy_mask = (q_range <= max_q.unsqueeze(1)) & (q_range >= 1)
        buy_mask &= allow_idx.unsqueeze(1)   # 套用持倉限制
        mask[0] = buy_mask

        sell_mask = (self.shares > 0) & (~baseline_mask)
        mask[1, sell_mask, 0] = True

        mask[2, 0, 0] = True
        return mask


    def _mark_to_market(self, prices: torch.Tensor) -> torch.Tensor:
        return (self.shares * prices).sum() + self.cash

    def _weights_vector(self, t: int) -> torch.Tensor:
        prices = self.prices_close[t]
        stock_val = (self.shares * prices).sum()
        V = stock_val + self.cash
        if V <= 0:
            w = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
            w[0] = 1.0
            return w
        return torch.cat([torch.tensor([self.cash / V], device=self.device),
                          (self.shares * prices) / V])

    def _slot_info(self, t: int) -> torch.Tensor:
        hold_info = []
        for s in range(self.max_holdings):
            i = self.slots[s]
            if i is None:
                hold_info.extend([0.0, 0.0])
            else:
                avg_cost = self.avg_costs[i]
                price = self.prices_close[t, i]
                floating_ret = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
                hold_info.extend([avg_cost, floating_ret])
        return torch.tensor(hold_info, dtype=torch.float32, device=self.device)

    def _make_obs(self, t: int) -> np.ndarray:
        feats = self.features[t - self.K + 1 : t + 1].reshape(-1).detach().cpu().numpy()
        portfolio = self._weights_vector(t).detach().cpu().numpy()
        slot_info = self._slot_info(t).detach().cpu().numpy()
        obs_vec = np.concatenate([feats, portfolio, slot_info], axis=0).astype(np.float32)
        return obs_vec

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        self._t = self.K
        self.cash = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.shares = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.avg_costs = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.slots = [None] * self.max_holdings
        self.portfolio_value = torch.tensor(float(self.initial_cash), dtype=torch.float32, device=self.device)
        self.trade_count = 0
        #  新增：初始化 peak_value
        self.peak_value = self.portfolio_value.clone()

        obs = self._make_obs(self._t)  # numpy array
        action_mask_3d = self._build_action_mask(self._t).detach().cpu().numpy()  # ✅ for AsyncVectorEnv pickling
        info = {"V": float(self.portfolio_value.item()), "action_mask_3d": action_mask_3d}
        return obs, info

    def step(self, action):
        if self._t + 1 >= self.T:
            obs_now = self._make_obs(self._t)
            info_end = {
                "msg": "no next day",
                "V": float(self.portfolio_value.item()),
                "action_mask_3d": self._build_action_mask(self._t).detach().cpu().numpy(),
            }
            return obs_now, 0.0, True, False, info_end

        t = self._t
        p_open  = self.prices_open[t + 1]
        p_close = self.prices_close[t + 1]

        op, idx, q = int(action[0]), int(action[1]), int(action[2])
        side, exec_shares, gross_cash, fees_tax = "HOLD", 0, 0, 0
        mask = self._build_action_mask(t)

        # BUY
        if op == 0 and 0 <= idx < self.N and 1 <= q <= self.QMAX and mask[0, idx, q]:
            price = float(p_open[idx])
            if price >= 10:
                lots  = min(int(q), int(self.cash // (self.lot_size * price * (1 + self.fee_buy))))
                if lots >= 1 and price > 0:
                    shares = lots * self.lot_size
                    gross  = int(round(shares * price))
                    fee    = int(round(gross * self.fee_buy))
                    cash_out = gross + fee
                    if cash_out <= self.cash + 1e-6:
                        old_shares = self.shares[idx]
                        is_new_pos = (old_shares <= 0)

                        if is_new_pos:
                            # 檢查 max_holdings
                            held_count = int(((self.shares > 0) & (~self.baseline_mask)).sum().item())
                            if held_count >= self.max_holdings:
                                # 滿倉 → 當作 HOLD，什麼都不做
                                pass
                            else:
                                # 開新倉
                                self.shares[idx] += shares
                                self.cash -= cash_out
                                self.avg_costs[idx] = price
                                for s in range(self.max_holdings):
                                    if self.slots[s] is None:
                                        self.slots[s] = idx
                                        break   # ✅ 一定要 break，避免同一股票塞進多個 slot
                                side, exec_shares, gross_cash, fees_tax = "BUY", shares, -gross, fee
                        else:
                            # 已持有該檔 → 加碼
                            self.shares[idx] += shares
                            self.cash -= cash_out
                            old_cost = self.avg_costs[idx]
                            self.avg_costs[idx] = (old_cost * old_shares + price * shares) / (old_shares + shares)
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

                # --- Debug: 賣之前快照
                sum_before  = float(self.shares.sum().item())
                this_before = float(self.shares[idx].item())

                # 清倉該檔
                self.shares[idx] = 0.0
                self.cash += cash_in
                self.avg_costs[idx] = 0.0

                # 清掉所有 slot 中的 idx
                for s in range(self.max_holdings):
                    if self.slots[s] == idx:
                        self.slots[s] = None

                side, exec_shares, gross_cash, fees_tax = "SELL_ALL", -shares, gross, fee + tax
                self.trade_count += 1

        # Reward
        reward, reward_info = self.reward_fn(self, action, side, p_close, t)

        # 更新 t
        self._t += 1
        terminated = (self._t + 1 >= self.T)
        self.portfolio_value = self._mark_to_market(self.prices_close[self._t])

        obs = self._make_obs(self._t)
        next_mask = self._build_action_mask(self._t)

        holdings_detail = {
            int(s): {
                "stock_id": str(self.ids[i]) if i is not None else None,
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
            "V": float(self.portfolio_value.item()),
            "date": str(pd.Timestamp(self.dates[self._t]).date()),
            "side": str(side),
            "stock_id": (self.ids[idx] if 0 <= idx < self.N else None),
            "lots": int(abs(exec_shares) // self.lot_size),
            "exec_shares": int(exec_shares),
            "gross_cash": float(gross_cash),
            "fees_tax": float(fees_tax),
            "cash": float(self.cash.item()),
            "held": int((self.shares > 0).sum().item()),
            "price": float(price) if "price" in locals() else None,
            "action_mask_3d": next_mask.detach().cpu().numpy(),
            "trade_count": int(self.trade_count),
            "slots_mapping": json.dumps({s: (self.ids[i] if i is not None else None) for s, i in enumerate(self.slots)}),
            "holdings_detail": json.dumps(holdings_detail),
            **{
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in reward_info.items()
            },
        }
        return obs, float(reward), terminated, False, info

