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
        lookback: int = 20, #決定往回看幾天來判斷今天的狀況(可能先抓20接近一個月 可調整)
        initial_cash: int = 100000,
        max_holdings: int = None,
        fee_buy: float = 0.001425 * 0.6,  # 台股手續費 0.1425% * 折扣
        fee_sell: float = 0.001425 * 0.6,
        tax_sell: float = 0.003,          # 交易稅 0.3%
        reward_mode: str = "Total_Profit",
        action_mode: str = "weights",     # 每次只能拿總資產的25 50 75 100%的其中一些來買? 總資產包括現在持有的股票 但不能超過現金餘額
        seed: int = 42,
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
        if mode == "daily_return":
            return float((V_new - V_prev) / max(1e-9, V_prev))
        elif mode == "total_profit":
            return float(V_new - self.initial_cash)
        else: #預設總報酬
            return float(V_new - self.initial_cash)
    
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
        return obs
    
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
        if self._t + 1 >= self.num_days:
            return self._make_obs(self._t), 0.0, True, False, {"msg": "no next day"}

        idx, lvl = int(action[0]), int(action[1])
        levels = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
        target_w = float(levels[lvl])

        # --- t+1 開盤成交 ---
        prices_open = self._prices(self._t + 1, kind="open")
        p = float(prices_open[idx])

        V_open = self._mark_to_market(prices_open)
        curr_val_i = float(self.shares[idx]) * p
        tgt_val_i  = float(target_w) * float(V_open)
        delta_val  = tgt_val_i - curr_val_i

        exec_shares, gross_cash, fees_tax = 0, 0, 0

        if delta_val > 1e-9 and p > 0 and self.cash > 0:
            # -------- BUY --------
            # 檢查 max_holdings 限制
            if self.max_holdings is not None and self.max_holdings > 0:
                held_count = int((self.shares > 0).sum())     # 目前已持有幾檔
                already_holding = self.shares[idx] > 0        # 這檔是否已經持有
                if held_count >= self.max_holdings and not already_holding:
                    # 已達上限且這檔沒持有 → 禁止買入，但不要 return
                    exec_shares, gross_cash, fees_tax = 0, 0, 0
                else:
                    max_by_cash = int(self.cash // ((1.0 + self.fee_buy) * p))
                    need_shares = int(delta_val // p)
                    raw_shares = min(max_by_cash, need_shares)
                    buy_shares = self._apply_lot_rules(raw_shares)

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
                #  沒持股卻要賣 → 當 HOLD 處理
                exec_shares, gross_cash, fees_tax = 0, 0, 0
            else:
                # -------- SELL --------
                hold = int(self.shares[idx])
                need_sell = int((-delta_val) // p)
                raw_sell = min(hold, need_sell)
                sell_shares = self._apply_lot_rules(raw_sell)

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


        # --- 收盤估值 & Reward = Total Profit ---
        prices_close = self._prices(self._t + 1, kind="close")
        V_prev = self.portfolio_value
        V_new = self._mark_to_market(prices_close)
        self._prev_portfolio_value = self.portfolio_value
        self.portfolio_value = V_new

        reward = self._compute_reward(V_prev, V_new) #根據reward 的不同計算方法選擇

        # 時間前進
        self._t += 1
        terminated = (self._t + 1 >= self.num_days)

        obs = self._make_obs(self._t)

        # 準備補充分步紀錄所需資訊
        curr_date = self.dates[self._t]                 # 已經前進過 self._t += 1，所以現在是當日（對應 t）
        stock_id = self.ids[idx]
        price_o = float(prices_open[idx])
        price_c = float(prices_close[idx])

        # 額外的權重資訊（方便日後分析）
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
            "exec_shares": int(exec_shares),   # 買正/賣負
            "gross_cash": int(gross_cash),     # 毛額（整數元）
            "fees_tax": int(fees_tax),         # 費稅（整數元）

            # ===== 新增：逐步紀錄需要的欄位 =====
            "date": str(pd.Timestamp(curr_date).date()),  # "YYYY-MM-DD"
            "stock_id": stock_id,
            "price_open": price_o,
            "price_close": price_c,
            "cash": int(self.cash),
            "shares_after": int(self.shares[idx]),
            "stock_value": stock_value,  
            "side": ("BUY" if exec_shares > 0 else ("SELL" if exec_shares < 0 else "HOLD")),
            "notional": int(abs(gross_cash)),   # 成交毛額（正數）
            # 額外觀察用
            "cash_w": cash_w,
            "stock_w_sum": stock_w_sum,
            "held": held,
            "w_max": w_max,
        }
        return obs, reward, terminated, False, info
