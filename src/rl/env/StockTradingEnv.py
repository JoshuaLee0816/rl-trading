import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
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

    metadata = {"render_modes": ["human"]}

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
        action_mode: str = "discrete"
    ):
        super().__init__()

        # baseline 0050 (不能交易，只能計算報酬)
        df_baseline = df[df["stock_id"] == "0050"].copy()
        if df_baseline.empty:
            raise ValueError("df 缺少 baseline stock_id = 0050 ")
        df_baseline = df_baseline.sort_values("date")
        self.baseline_close = df_baseline["close"].to_numpy(dtype=np.float32)

        # 初始化參數
        self.ids = list(stock_ids)
        self.N = len(self.ids)
        self.K = int(lookback)
        self.initial_cash = float(initial_cash)
        self.max_holdings = int(max_holdings)
        self.QMAX = int(qmax_per_trade)
        self.fee_buy, self.fee_sell, self.tax_sell = float(fee_buy), float(fee_sell), float(tax_sell)
        self.lot_size = int(lot_size)
        self.rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode

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

        # convert df's "open" and "close" into 2-dimensions matrix
        open_pv  = df.pivot(index="date", columns="stock_id", values="open").reindex(index=self.dates, columns=self.ids)
        close_pv = df.pivot(index="date", columns="stock_id", values="close").reindex(index=self.dates, columns=self.ids)
        self.prices_open  = open_pv.to_numpy(dtype=np.float64)   # [T, N]
        self.prices_close = close_pv.to_numpy(dtype=np.float64)  # [T, N]

        # 全特徵 3D -> numpy 
        feat_mats = []     #專門存放每個特徵轉換後的矩陣
        for c in self._feat_cols:
            pv = df.pivot(index="date", columns="stock_id", values=c).reindex(index=self.dates, columns=self.ids)
            feat_mats.append(pv.to_numpy(dtype=np.float32)[..., None])   # [T, N, 1]
        self.features = np.concatenate(feat_mats, axis=2) if feat_mats else np.zeros((self.T, self.N, 0))

        # 狀態 (用來描述當前狀態的變數)
        self._t = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self.avg_costs = None
        self.slots = None  # slot-based 持倉紀錄

        # 定義環境的觀測空間
        obs_dim = self.N * self.features.shape[2] * self.K + (1 + self.N) + 2 * self.max_holdings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 定義動作空間
        self.action_space = spaces.MultiDiscrete([3, self.N, self.QMAX + 1])

    # endregion 初始化部分

    # region 小工具部分
    def _build_action_mask(self, t: int) -> np.ndarray:
        """
        回傳形狀 (3, N, QMAX+1) 的布林遮罩
        """
        mask = np.zeros((3, self.N, self.QMAX + 1), dtype=bool)
        if t + 1 >= self.T:
            mask[2, 0, 0] = True
            return mask

        p_open = self.prices_open[t + 1]
        baseline_mask = np.array([sid == "0050" for sid in self.ids])

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

    def _mark_to_market(self, prices: np.ndarray) -> float:
        return float((self.shares * prices).sum() + self.cash)

    def _weights_vector(self, t: int) -> np.ndarray:
        prices = self.prices_close[t]
        stock_val = float((self.shares * prices).sum())
        V = stock_val + float(self.cash)
        if V <= 0:
            w = np.zeros(self.N + 1, dtype=np.float64)
            w[0] = 1.0
            return w
        return np.concatenate([[self.cash / V], (self.shares * prices) / V])

    def _features_window(self, t: int) -> np.ndarray:
        win = self.features[t - self.K + 1 : t + 1]   # [K, N, F]
        return win.reshape(-1)

    def _slot_info(self, t: int) -> np.ndarray:
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
        return np.array(hold_info, dtype=np.float32)

    def _make_obs(self, t: int) -> np.ndarray:
        feats = self._features_window(t)
        weights = self._weights_vector(t)
        slot_info = self._slot_info(t)
        obs = np.concatenate([feats, weights, slot_info]).astype(np.float32)
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return np.clip(obs, -1e6, 1e6)
    # endregion 小工具部分

    # region GymAPI
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.K
        self.cash = float(self.initial_cash)
        self.shares = np.zeros(self.N, dtype=np.int64)
        self.avg_costs = np.zeros(self.N, dtype=np.float64)
        self.slots = [None] * self.max_holdings
        self.portfolio_value = float(self.initial_cash)
        self.trade_count = 0

        obs = self._make_obs(self._t)
        action_mask_3d = self._build_action_mask(self._t)

        info = {"V": int(self.portfolio_value), "action_mask_3d": action_mask_3d}
        return obs, info

    def step(self, action):
        if self._t + 1 >= self.T:
            obs_now = self._make_obs(self._t)
            info_end = {"msg": "no next day", "V": int(self.portfolio_value),
                        "action_mask_3d": self._build_action_mask(self._t)}
            return obs_now, 0.0, True, False, info_end

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

        # Reward
        V_prev = float(self.portfolio_value)
        V_new  = float(self._mark_to_market(p_close))
        self.portfolio_value = V_new
        portfolio_return = float(np.log(max(V_new, 1e-12) / max(V_prev, 1e-12)))
        baseline_return  = float(np.log(max(self.baseline_close[t + 1], 1e-12) /
                                        max(self.baseline_close[t], 1e-12)))
        reward = portfolio_return - baseline_return
        if side == "BUY" & "SELL":
            reward -= 0.00005

        # === 個股懲罰 (slot-based) ===
        penalty = 0.0
        for s, i in enumerate(self.slots):
            if i is not None and self.avg_costs[i] > 0:
                cur_price = self.prices_close[self._t, i]
                floating_ret = (cur_price - self.avg_costs[i]) / self.avg_costs[i]
                if floating_ret < 0:
                    # 線性懲罰
                    # penalty += abs(floating_ret) * 0.05   # α=0.05，可調
                    # 指數型懲罰
                    penalty += (np.exp(-5 * floating_ret) - 1)*0.1
        reward -= penalty

        self._t += 1
        terminated = (self._t + 1 >= self.T)
        obs = self._make_obs(self._t)
        next_mask = self._build_action_mask(self._t)

        # holdings_detail
        holdings_detail = {
            s: {
                "stock_id": self.ids[i] if i is not None else None,
                "shares": int(self.shares[i]) if i is not None else 0,
                "avg_cost": float(self.avg_costs[i]) if i is not None else 0.0,
                "cur_price": float(self.prices_close[self._t, i]) if i is not None else 0.0,
                "floating_ret": (
                    (self.prices_close[self._t, i] - self.avg_costs[i]) / self.avg_costs[i]
                    if (i is not None and self.avg_costs[i] > 0) else 0.0
                )
            }
            for s, i in enumerate(self.slots)
        }

        info = {
            "V": int(self.portfolio_value),
            "date": str(pd.Timestamp(self.dates[self._t]).date()),
            "side": side,
            "stock_id": (self.ids[idx] if 0 <= idx < self.N else None),
            "lots": (abs(exec_shares) // self.lot_size),
            "exec_shares": int(exec_shares),
            "gross_cash": int(gross_cash),
            "fees_tax": int(fees_tax),
            "cash": int(round(self.cash)),
            "held": int((self.shares > 0).sum()),
            "action_mask_3d": next_mask,
            "baseline_return": baseline_return,
            "trade_count": self.trade_count,
            "slots_mapping": {s: (self.ids[i] if i is not None else None) for s, i in enumerate(self.slots)},
            "holdings_detail": holdings_detail,   
        }

        return obs, reward, terminated, False, info
