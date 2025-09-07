"""
StockTradingEnv 設計目標
------------------------
環境內部 (State):
- 記錄目前總資產 V_t (現金 + 股票市值)
- 記錄每檔股票的持股數量
- 記錄現金餘額
- 當前時間索引 (t)

Observation (觀測):
- 市場特徵 (各股票價量、技術指標等, lookback 視窗)
- 投組狀態 (現金佔比、各股倉位佔比)

Action (動作):
- 在 t 日決策 t+1 開盤成交

Reward (報酬):
- 採用「總資產日變動率」:
- 已內含交易成本

流程:
1. t 日觀察資料與投組狀態
2. Agent 輸入動作 (action)
3. t+1 開盤價執行調整 (扣除成本)
4. t+1 收盤價重新計值資產 V_t
5. 計算 reward 回饋給 agent
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    metadata = {"render_modes":["human"]} #環境渲染模式

    def __init__(
        self,
        df: pd.DataFrame,
        stock_ids,
        lookback: int,
        initial_cash: int,
        max_holdings: int,
        reward_mode: str,        # 從 config 傳進來
        action_mode: str,        # 從 config 傳進來
        seed: int = 42,
        fee_buy: float = 0.001425 * 0.6,
        fee_sell: float = 0.001425 * 0.6,
        tax_sell: float = 0.003,
    ):
        super().__init__() #呼叫gym.Env 父類別屬性初始化

        #Inputs Config
        self.df = (df.copy()
                   .sort_values(["date", "stock_id"])
                   .reset_index(drop=True)) #丟掉舊的index重新建立新的
        self.ids = list(stock_ids)
        self.N = len(self.ids)
        self.K = int(lookback)
        self.initial_cash = int(initial_cash)
        self.max_holdings = max_holdings
        self.fee_buy, self.fee_sell, self.tax_sell = float(fee_buy), float(fee_sell), float(tax_sell)
        self.reward_mode = str(reward_mode)
        self.action_mode = str(action_mode)
        self.rng = np.random.default_rng(seed)
        self.lot_size = 1000 #交易單位先暫訂為1張，不支援零股
        self.min_trade_value = 10000   # 可選：為避免一直小額來回交易，設一個金額閾值（台幣）
        
        #Basic validataions
        #資料要先從寬表轉成長表 格式如下
        # date        stock_id   open   high   low   close  MA... dividend等等
        # 2015-07-07     1216    54.1   54.7   54.1   54.2
        # 2015-07-07     1303    70.4   71.0   69.8   70.0
        if not {"date", "stock_id", "open", "close"}.issubset(self.df.columns):
            raise ValueError("df 必須至少包含: ['date','stock_id','open','close']")
        if not np.issubdtype(self.df["date"].dtype, np.datetime64):
            self.df["date"] = pd.to_datetime(self.df["date"])

        missing = set(self.ids) - set(self.df["stock_id"].unique())
        if missing:
            raise ValueError(f"df 缺少以下 stock_ids: {sorted(missing)}")
        
        # 僅保留選定股票池
        self.df = self.df[self.df["stock_id"].isin(self.ids)]

        # ---- Build date index ----
        self.dates = np.sort(self.df["date"].unique())
        self.num_days = len(self.dates)
        if self.num_days <= self.K + 1:
            raise ValueError("資料天數不足（需要 > lookback+1 天）。")

        # ==== 預先轉成 NumPy，避免 step() 反覆 groupby/pivot（加速關鍵） ====
        self._feat_cols = [c for c in self.df.columns if c not in ["date", "stock_id"]]

        _close_pv = (self.df.pivot(index="date", columns="stock_id", values="close")
                            .reindex(index=self.dates)
                            .reindex(columns=self.ids))
        _open_pv  = (self.df.pivot(index="date", columns="stock_id", values="open")
                            .reindex(index=self.dates)
                            .reindex(columns=self.ids))
        self.prices_close = _close_pv.to_numpy(dtype=np.float64)   # [T, N]
        self.prices_open  = _open_pv.to_numpy(dtype=np.float64)    # [T, N]

        _feat_mats = []
        for c in self._feat_cols:
            _pv = (self.df.pivot(index="date", columns="stock_id", values=c)
                         .reindex(index=self.dates)
                         .reindex(columns=self.ids))
            _feat_mats.append(_pv.to_numpy(dtype=np.float64)[..., None])  # [T,N,1]
        self.features = np.concatenate(_feat_mats, axis=2) if _feat_mats else \
                        np.zeros((len(self.dates), self.N, 0), dtype=np.float64)  # [T,N,F]
        
        # ---- State placeholders（reset 時會填）----
        self._t = None                   # 目前時間
        self.cash = None                 # 現金餘額
        self.shares = None               # shape (N,), 各股持股數
        self.portfolio_value = None      # 當前總資產 V_t
        self._prev_portfolio_value = None

        #建立observation_space, action_space
        feat_dim = self._infer_feat_dim()   # = N * K
        obs_dim = feat_dim + (1 +self.N)    # feat_dim再加上現金跟每個股票的存股 observation = [features + cash + shares]
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape = (obs_dim,), dtype=np.float32
        )
        if self.action_mode == "weights":
            # MultiDiscrete([N, 4])：
            # - 第 1 維：股票索引 (0 ~ N-1)
            # - 第 2 維：目標比例等級 (0~3 對應 [25%, 50%, 75%, 100%])
            self.action_space = spaces.MultiDiscrete([self.N, 5])  # 改為含0%，所以5個等級

        elif self.action_mode == "portfolio":
            self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32
        )
        else:
            raise NotImplementedError("目前先支援 action_mode='weights'")

    """
    小工具的部分
    """

    def _round_currency(self, x: float) -> int:
        return int(round(x))

    def _apply_lot_rules(self, desired_shares: int) -> int:
        """只允許整張：向下取整到 1000 的倍數"""
        return (desired_shares // self.lot_size) * self.lot_size
    
    def _prices(self, t: int, kind: str = "close") -> np.ndarray:
        # 直接用預先好的矩陣存取（O(1) 切片）
        if kind == "close":
            return self.prices_close[t]
        elif kind == "open":
            return self.prices_open[t]
        else:
            raise ValueError("kind 必須是 'close' 或 'open'")

    def _mark_to_market(self, prices: np.ndarray) -> int:
        stock_val = float((self.shares * prices).sum())
        total = self._round_currency(stock_val) + int(self.cash)
        return int(total)
    
    def _compute_reward(self, V_prev: float, V_new: float) -> float:
        mode = self.reward_mode.lower()
        if mode == "daily_return": #還是要一週或是一月的平均return?
            return float((V_new - V_prev) / max(1e-9, V_prev))
        elif mode == "total_profit":
            return float(V_new - self.initial_cash)
        else: #預設總報酬
            return float(V_new - self.initial_cash)
    
    def _normalize_portfolio(self, a: np.ndarray) -> np.ndarray:
        """
        將動作向量 a (N+1,) 轉成合法資產配置向量 w：非負且總和=1
        """
        a = np.asarray(a, dtype=np.float32).reshape(self.N + 1)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # 只允許非負
        a = np.maximum(a, 0.0)

        s = float(a.sum())
        if s <= 1e-12:
            # 若全 0，預設全放現金
            w = np.zeros(self.N + 1, dtype=np.float32)
            w[-1] = 1.0
            return w
        return (a / s).astype(np.float32)

    def _alloc_to_target_shares(self, weights: np.ndarray, prices: np.ndarray,
                                portfolio_value: float) -> tuple[np.ndarray, float]:
        """
        將資產配置 weights (N+1,) 轉為目標股數 (N,) 與目標現金(含未能整股用掉的零頭)
        weights[:-1] -> 各股比例，weights[-1] -> 現金比例
        """
        w_stock = weights[:-1]                         # (N,)
        w_cash  = float(weights[-1])

        # 目標持股市值（金額）
        target_stock_value = w_stock * portfolio_value # (N,)
        # 以收盤價計算目標股數（整股、考慮 lot_size）
        target_shares = np.floor(
            np.maximum(0.0, target_stock_value) / np.maximum(prices, 1e-9) / self.lot_size
        ).astype(np.int64) * self.lot_size            # (N,)

        # 實際佔用的金額（四捨五入後會略小於或等於 target_stock_value）
        used_value = (target_shares * prices).sum()
        target_cash = max(0.0, portfolio_value - used_value)

        # 若你希望嚴格讓現金≈w_cash*PV，可在這裡微調（通常不必）
        return target_shares, target_cash

    # --- lot-size helper: 統一整股規則 ---
    def _apply_lot(self, shares: int) -> int:
        """
        將 shares 依 lot_size 做整股化。
        - 若已有 self._apply_lot_rules()，優先使用原規則（相容舊邏輯）。
        - 否則用 lot_size 進行向下取整。
        """
        # 基本清理
        s = int(np.floor(float(shares)))
        lot = int(getattr(self, "lot_size", 1) or 1)
        s = max(0, s)

        # 如果原本就有 _apply_lot_rules，就用原本規則
        if hasattr(self, "_apply_lot_rules") and callable(self._apply_lot_rules):
            try:
                return int(self._apply_lot_rules(s))
            except Exception:
                pass  # 保險，fallback 到簡單整股

        # 簡單整股：向下取到 lot 的倍數
        if lot > 1:
            s = (s // lot) * lot
        return int(s)

    
    """
    小工具結束的部分
    """
    #STEP 2 : observation and reset

    def reset(self, *, seed=None, options=None):
        #Gymnasium定義必須回傳(observation, info)
        super().reset(seed=seed)

        #初始資產與倉位
        self._t = self.K
        self.cash = int(self.initial_cash)
        self.shares = np.zeros(self.N, dtype = np.int64)
        self._prev_portfolio_value = self.initial_cash

        #以t日收盤估值 (此時尚未持股，所以剛好等於現金)
        self.portfolio_value = int(self.cash)
        obs = self._make_obs(self._t)
        info = {"V": self.portfolio_value}
        return obs, info
    
    def _make_obs(self, t:int) -> np.ndarray:
        feats = self._features_window(t)    # shape = (N*K,)
        weights = self._weights_vector(t)   # 現金佔比 + N股佔比
        obs = np.concatenate([feats, weights], axis = 0).astype(np.float32)

        # === 方法2：標準化 (建議) ===
        mean = np.mean(obs)
        std = np.std(obs) + 1e-6
        obs = (obs - mean) / std

        # 避免極端值
        obs = np.clip(obs, -10, 10)

        return obs/1000
    
    def _infer_feat_dim(self) -> int:
        # 除了 date, stock_id 之外的所有數值欄位都算特徵 (千萬不能取道date, and stock_id當作特徵)
        return self.N * len(self._feat_cols) * self.K

    def _features_window(self, t:int) -> np.ndarray:
        #取t 往回K天的
        start = t - self.K + 1
        end   = t + 1
        win = self.features[start:end]   # [K, N, F]
        return win.reshape(-1, order="C")   # shape (K*N*F,)
    
    def _weights_vector(self, t: int) -> np.ndarray:
        #回傳現金佔比 + 各股市值佔比（用 t 日收盤估值）
        prices = self.prices_close[t]  # [N]
        stock_val = (self.shares * prices).sum()
        V = stock_val + self.cash
        if V <= 0:
            vec = np.zeros(self.N + 1, dtype=np.float64)
            vec[0] = 1.0
            return vec
        weights_stock = (self.shares * prices) / V           # N 維
        cash_w = self.cash / V
        return np.concatenate([[cash_w], weights_stock]).astype(np.float64)
    
    #STEP 3 : Step
    def step(self, action):
        # 沒有明天
        if self._t + 1 >= self.num_days:
            return self._make_obs(self._t), 0.0, True, False, {"msg": "no next day"}

        # 先抓 t+1 開盤/收盤價，估值等共用資料
        prices_open  = self._prices(self._t + 1, kind="open").astype(np.float64)
        prices_close = self._prices(self._t + 1, kind="close").astype(np.float64)

        # ====== 分支 1：Portfolio 模式（PPO 用） ======  
        if self.action_mode == "portfolio":
            # action: [w_stock(1..N), w_cash]，先正規化為總和=1
            w = self._normalize_portfolio(action)  # shape (N+1,)

            # 以 t+1 開盤價、目前總資產 V_open 為基準，計算目標股數
            V_open = float(self._mark_to_market(prices_open))
            w_stock = w[:-1]                       # 長度 N
            # 目標金額
            tgt_values = (w_stock * V_open)        # N,
            # 轉成目標股數（整股）
            tgt_shares = np.floor(
                np.maximum(0.0, tgt_values) / np.maximum(prices_open, 1e-9) / max(1, self.lot_size)
            ).astype(np.int64) * max(1, self.lot_size)

            # 交易差額
            delta = tgt_shares - self.shares.astype(np.int64)

            # 先賣後買，避免現金不足
            exec_shares_total = 0
            gross_cash_total  = 0
            fees_tax_total    = 0

            # -------- SELL（先處理負的）--------
            for i in np.where(delta < 0)[0]:
                p = float(prices_open[i])
                if p <= 0: 
                    continue
                need = int(-delta[i])
                hold = int(self.shares[i])
                raw  = min(hold, need)
                sell_shares = self._apply_lot(raw)
                if sell_shares <= 0:
                    continue

                gross = self._round_currency(sell_shares * p)
                fee   = self._round_currency(gross * self.fee_sell)
                tax   = self._round_currency(gross * self.tax_sell)
                cash_in = gross - fee - tax

                self.shares[i] -= sell_shares
                self.cash      += cash_in

                exec_shares_total += -sell_shares
                gross_cash_total  += gross
                fees_tax_total    += fee + tax

            # -------- BUY（再處理正的）--------
            for i in np.where(delta > 0)[0]:
                p = float(prices_open[i])
                if p <= 0:
                    continue
                need = int(delta[i])

                # 依現金上限計算可買股數
                max_by_cash = int(self.cash // ((1.0 + self.fee_buy) * p))
                raw = min(need, max_by_cash)
                buy_shares = self._apply_lot(raw)
                if buy_shares <= 0:
                    continue

                gross = self._round_currency(buy_shares * p)
                fee   = self._round_currency(gross * self.fee_buy)
                cash_out = gross + fee
                if cash_out > self.cash:
                    continue  # 安全檢查：現金不足就跳過（或可再縮單）

                self.shares[i] += buy_shares
                self.cash      -= cash_out

                exec_shares_total += buy_shares
                gross_cash_total  += -gross
                fees_tax_total    += fee

            # --- 收盤估值 & Reward = Total Profit（沿用你的做法） ---
            V_prev = self.portfolio_value
            V_new  = float(self._mark_to_market(prices_close))
            self._prev_portfolio_value = self.portfolio_value
            self.portfolio_value = V_new

            reward = float(self._compute_reward(V_prev, V_new))

            # 時間前進
            self._t += 1
            terminated = (self._t + 1 >= self.num_days)
            obs = self._make_obs(self._t)

            # === info ===
            curr_date = self.dates[self._t]
            prices_c_now = self.prices_close[self._t]
            V_now = max(1e-9, (self.shares * prices_c_now).sum() + self.cash)
            w_vec = self._weights_vector(self._t)  # [cash_w, w1, w2, ...]
            cash_w = float(w_vec[0])
            stock_w_sum = float(w_vec[1:].sum())
            held = int((self.shares > 0).sum())
            w_stocks_now = (self.shares * prices_c_now) / V_now if V_now > 0 else np.zeros(self.N)
            w_max = float(w_stocks_now.max()) if held > 0 else 0.0
            stock_value = int((self.shares * prices_c_now).sum())

            # 兼容原本欄位：picked_idx/stock_id 若是 portfolio，就給 -1 / "PORTFOLIO"
            info = {
                "V": int(self.portfolio_value),
                "picked_idx": -1,                      # ### NEW: 兼容欄位
                "target_w": float(stock_w_sum),        # ### NEW: 這裡給股票總權重
                "exec_shares": int(exec_shares_total),
                "gross_cash": int(gross_cash_total),
                "fees_tax": int(fees_tax_total),

                "date": str(pd.Timestamp(curr_date).date()),
                "stock_id": "PORTFOLIO",               # ### NEW
                "price_open": float(np.nan),           # ### NEW
                "price_close": float(np.nan),          # ### NEW
                "cash": int(self.cash),
                "shares_after": int(self.shares.sum()),# ### NEW: 總股數
                "stock_value": stock_value,
                "side": "MIXED" if exec_shares_total != 0 else "HOLD",
                "notional": int(abs(gross_cash_total)),
                "cash_w": cash_w,
                "stock_w_sum": stock_w_sum,
                "held": held,
                "w_max": w_max,

                # ### 方便追蹤：本步實際的目標配置（股票+現金）
                "target_weights": w.astype(float).tolist(),
            }
            return obs, reward, terminated, False, info

        # ====== 分支 2：DQN 模式（沿用你原本的程式） ======
        idx, lvl = int(action[0]), int(action[1])
        levels = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
        target_w = float(levels[lvl])

        p = float(prices_open[idx])
        V_open = float(self._mark_to_market(prices_open))
        curr_val_i = float(self.shares[idx]) * p
        tgt_val_i  = float(target_w) * float(V_open)
        delta_val  = tgt_val_i - curr_val_i

        exec_shares, gross_cash, fees_tax = 0, 0, 0

        if delta_val > 1e-9 and p > 0 and self.cash > 0:
            # BUY
            if self.max_holdings is not None and self.max_holdings > 0:
                held_count = int((self.shares > 0).sum())
                already_holding = self.shares[idx] > 0
                if held_count >= self.max_holdings and not already_holding:
                    exec_shares, gross_cash, fees_tax = 0, 0, 0
                else:
                    max_by_cash = int(self.cash // ((1.0 + self.fee_buy) * p))
                    need_shares = int(delta_val // p)
                    raw_shares = min(max_by_cash, need_shares)
                    buy_shares = self._apply_lot(raw_shares)

                    if buy_shares > 0:
                        gross = self._round_currency(buy_shares * p)
                        fee   = self._round_currency(gross * self.fee_buy)
                        cash_out = gross + fee
                        if cash_out <= self.cash:
                            self.shares[idx] += buy_shares
                            self.cash        -= cash_out
                            exec_shares = buy_shares
                            gross_cash  = -gross
                            fees_tax    = fee

        elif delta_val < -1e-9 and p > 0:
            if self.shares[idx] <= 0:
                exec_shares, gross_cash, fees_tax = 0, 0, 0
            else:
                # SELL
                hold = int(self.shares[idx])
                need_sell = int((-delta_val) // p)
                raw_sell = min(hold, need_sell)
                sell_shares = self._apply_lot(raw_sell)

                if sell_shares > 0:
                    gross = self._round_currency(sell_shares * p)
                    fee   = self._round_currency(gross * self.fee_sell)
                    tax   = self._round_currency(gross * self.tax_sell)
                    cash_in = gross - fee - tax
                    self.shares[idx] -= sell_shares
                    self.cash        += cash_in
                    exec_shares = -sell_shares
                    gross_cash  = gross
                    fees_tax    = fee + tax

        # --- 收盤估值 & Reward ---
        V_prev = self.portfolio_value
        V_new  = float(self._mark_to_market(prices_close))
        self._prev_portfolio_value = self.portfolio_value
        self.portfolio_value = V_new

        reward = float(self._compute_reward(V_prev, V_new))

        # 時間前進
        self._t += 1
        terminated = (self._t + 1 >= self.num_days)
        obs = self._make_obs(self._t)

        # 補充 info（保持你的原格式）
        curr_date = self.dates[self._t]
        stock_id = self.ids[idx]
        price_o = float(prices_open[idx])
        price_c = float(prices_close[idx])

        w_vec = self._weights_vector(self._t)           # [cash_w, w1, w2, ...]
        cash_w = float(w_vec[0])
        stock_w_sum = float(w_vec[1:].sum())
        prices_c_now = self.prices_close[self._t]
        V_now = max(1e-9, (self.shares * prices_c_now).sum() + self.cash)
        w_stocks_now = (self.shares * prices_c_now) / V_now if V_now > 0 else np.zeros(self.N)
        held = int((self.shares > 0).sum())
        w_max = float(w_stocks_now.max()) if held > 0 else 0.0
        stock_value = int((self.shares * prices_c_now).sum())

        info = {
            "V": int(self.portfolio_value),
            "picked_idx": idx,
            "target_w": target_w,
            "exec_shares": int(exec_shares),
            "gross_cash": int(gross_cash),
            "fees_tax": int(fees_tax),

            "date": str(pd.Timestamp(curr_date).date()),
            "stock_id": stock_id,
            "price_open": price_o,
            "price_close": price_c,
            "cash": int(self.cash),
            "shares_after": int(self.shares[idx]),
            "stock_value": stock_value,
            "side": ("BUY" if exec_shares > 0 else ("SELL" if exec_shares < 0 else "HOLD")),
            "notional": int(abs(gross_cash)),
            "cash_w": cash_w,
            "stock_w_sum": stock_w_sum,
            "held": held,
            "w_max": w_max,
        }
        return obs, reward, terminated, False, info
