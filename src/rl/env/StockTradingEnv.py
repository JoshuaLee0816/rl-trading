import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """
    StockTradingEnv (Broker-style, 1 trade per day)

    ▶ Observation (obs)
      shape = (N*F*K) + (1 + N)
      = [K天 * N檔 * F特徵攤平] + [現金佔比] + [各股市值佔比(用t日收盤估值)]
      調用時序：在 t 觀測 → t 決策 → t+1 開盤成交 → t+1 收盤估值

    ▶ Action (MultiDiscrete([3, N, QMAX+1]))
      op  ∈ {0,1,2} = {BUY, SELL_ALL, HOLD}
      idx ∈ {0..N-1}   股票索引
      q   ∈ {0..QMAX}  買入張數（單位=張，=1000股；僅 BUY 時有效，q>=1 才有意義）
      * 每天只會執行一筆（買或全賣或不動）
      * 不合法動作不懲罰，直接視為 HOLD（建議在 policy 端用 action_mask 做 masked softmax）

    ▶ Reward
      r_t = log( V_{t+1} / V_t )，以 t+1 收盤估值，已含交易成本/稅

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

        # convert df's "open" and "close" into 2-dimensions matrix 單獨先處理因為step()都會用到 常要提取
        """open_pv & close_pv (會產出兩個矩陣)
        stock_id     2330    2317    2454 ...
        date
        2020-01-02   330.0   87.5    290.0 ...
        2020-01-03   335.5   88.0    293.0 ...
        ...
        """

        open_pv  = df.pivot(index="date", columns="stock_id", values="open").reindex(index=self.dates, columns=self.ids)
        close_pv = df.pivot(index="date", columns="stock_id", values="close").reindex(index=self.dates, columns=self.ids)
        self.prices_open  = open_pv.to_numpy(dtype=np.float64)   # [T, N]
        self.prices_close = close_pv.to_numpy(dtype=np.float64)  # [T, N]

        # 全特徵 3D -> numpy 
        """
        每個特徵都會變成 [T *N]
        全部疊起來 變成 [T*N* features總數]
        """
        feat_mats = []     #專門存放每個特徵轉換後的矩陣
        for c in self._feat_cols:
            pv = df.pivot(index="date", columns="stock_id", values=c).reindex(index=self.dates, columns=self.ids)
            feat_mats.append(pv.to_numpy(dtype=np.float32)[..., None])   # [T, N, 1]
        self.features = np.concatenate(feat_mats, axis=2) if feat_mats else np.zeros((self.T, self.N, 0))

        # 狀態 (用來描述當前狀態的變數)
        """
        _t => 模擬到第幾天
        cash => 現金餘額多少
        shares => shares[i] 第i檔股票手上現在有幾股
        portfolio_value => total assets current
        """
        self._t = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None

        # 定義環境的觀測空間
        """
        [K *N *F] => observation是 "最近K天,N檔股票, 每檔有F個特徵" 攤平成一個向量
        """
        obs_dim = self.N * self.features.shape[2] * self.K + (1 + self.N)    #obs_dim是一個 一維向量
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 定義動作空間 三維 [操作類型, idx哪一檔, q數量] EX:[1,5,3]= 買入 第 5 檔股票 3 張
        """
        操作類型:
        0 = 不動作 (hold)
        1 = 買入 (buy)
        2 = 賣出 (sell)
        """
        self.action_space = spaces.MultiDiscrete([3, self.N, self.QMAX + 1])    #QMax +1 因為有可能有0

    # endregion 初始化部分

    # region 小工具部分
    def _mark_to_market(self, prices: np.ndarray) -> float:
        return float((self.shares * prices).sum() + self.cash)

    def _weights_vector(self, t: int) -> np.ndarray:
        # [現金佔比] + [各股市值佔比]（用 t 日收盤估值）
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

    def _make_obs(self, t: int) -> np.ndarray:
        feats = self._features_window(t)
        weights = self._weights_vector(t)
        obs = np.concatenate([feats, weights]).astype(np.float32)
        return np.clip(obs, -1e6, 1e6)

        # —— 合法化遮罩（不做門檻與懲罰） ——
    def _buy_mask(self, p_open: np.ndarray) -> np.ndarray:
        """
        回傳對每檔股票「可買至少 1 張」的布林向量 [N]
        - 現金足夠買 1 張（含手續費）
        - 若該檔尚未持有，需符合 max_holdings 名額（held_count < max_holdings）
        - 已持有者可加碼（不受名額限制）
        - 一天只會執行一個動作，通過遮罩也是候選，後續抽樣
        """
        cost_1lot = np.ceil(p_open * self.lot_size * (1 + self.fee_buy))  # [N]
        has_cash = cost_1lot <= (self.cash + 1e-6)
        if self.max_holdings is None or self.max_holdings <= 0:
            return has_cash
        held_flags = (self.shares > 0)
        space_left = (held_flags.sum() < self.max_holdings)
        return np.where(held_flags, has_cash, has_cash & space_left)

    def _max_affordable_lots(self, price_open: float) -> int:
        """依現金計算最多可買幾張（向下取整）"""
        if price_open <= 0:
            return 0
        cost_per_lot = self.lot_size * price_open * (1 + self.fee_buy)
        return int(np.floor(self.cash / max(cost_per_lot, 1e-9)))

    def _sell_mask(self) -> np.ndarray:
        """賣出合法性：有持股即可（一次全賣）"""
        return self.shares > 0

    def _build_action_mask(self, t: int) -> np.ndarray:
        """
        回傳形狀 (3, N, QMAX+1) 的布林遮罩：
          mask[0, i, q] -> BUY i, q 張 合法 (q>=1)
          mask[1, i, 0] -> SELL_ALL i 合法（q忽略，習慣上只開 q=0）
          mask[2, 0, 0] -> HOLD 合法
        其它位置 False
        """
        mask = np.zeros((3, self.N, self.QMAX + 1), dtype=bool)
        if t + 1 >= self.T:
            mask[2, 0, 0] = True  # 只有 HOLD
            return mask

        p_open = self.prices_open[t + 1]

        # BUY 面
        can_buy_any = self._buy_mask(p_open)  # [N]
        for i in range(self.N):
            if not can_buy_any[i] or p_open[i] <= 0:
                continue
            max_by_cash = self._max_affordable_lots(float(p_open[i]))
            max_q = max(0, min(self.QMAX, max_by_cash))
            if max_q >= 1:
                mask[0, i, 1:max_q + 1] = True  # q>=1 才有意義

        # SELL_ALL 面（q 忽略，僅 q=0）
        sellable = self._sell_mask()
        for i in range(self.N):
            if sellable[i]:
                mask[1, i, 0] = True

        # HOLD
        mask[2, 0, 0] = True
        return mask
    
    # endregion 小工具部分

    # region GymAPI
    """
    reset => 把環境歸零 準備新的episode
    Gymnasium標準API => args: seed, options
    * 必須用關鍵字指定，不能用位置參數
    """
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化時間與資產
        self._t = self.K
        self.cash = float(self.initial_cash)
        self.shares = np.zeros(self.N, dtype=np.int64)
        self.portfolio_value = float(self.initial_cash)

        # 產生初始觀測
        obs = self._make_obs(self._t)
        
        # mask 擋住不合法動作 (現金不足不能購買)
        action_mask_3d = self._build_action_mask(self._t)

        info = {
            "V": int(self.portfolio_value),  #回傳當前投資組合 total asset
            "action_mask_3d": action_mask_3d,   #回傳action mask的時候action 分布還沒有出來 只是單純的擋住不合法 分布要靠ppo agent產出?
        }
        return obs, info


    def step(self, action):

        # region 最後一天沒法走
        # 若沒有下一天就結束：仍回傳當前 obs，reward=0，並提供「終局遮罩」（只允許 HOLD）
        if self._t + 1 >= self.T:
            obs_now = self._make_obs(self._t)
            info_end = {
                "msg": "no next day",
                "V": int(self.portfolio_value),
                "action_mask_3d": self._build_action_mask(self._t), #最後一天只有HOLD是合法動作
            }
            return obs_now, 0.0, True, False, info_end #reward 0.0因為沒有下一天, terminated = True
        # endregion 最後一天沒法走

        t = self._t
        p_open  = self.prices_open[t + 1]   # t+1 開盤成交
        p_close = self.prices_close[t + 1]  # t+1 收盤估值

        # 把agent傳進來的動作向量action 拆成三個部分 op, idx, q
        """
        Agent 已經決定好要做哪一檔的哪一個動作了
        """
        op, idx, q = int(action[0]), int(action[1]), int(action[2])

        # 預設不動 (還沒有合法Mask前先不動)
        side, exec_shares, gross_cash, fees_tax = "HOLD", 0, 0, 0

        # 讀取合法遮罩；若動作不合法 視為 HOLD
        """
        目前設定為每次只做一個動作，所以如果被遮罩掉了 就是沒有動作=HOLD (可以之後改成不只一個動作(候選動作)或是把初始資金拉高(不建議))
        """
        mask = self._build_action_mask(t)

        if op == 0 and 0 <= idx < self.N and 1 <= q <= self.QMAX and mask[0, idx, q]: 
            """
            mask確定合法True的動作才會進入,且至少買一張不超過上限qmax
            """
            # BUY: q 張（一次只交易這一檔）
            price = float(p_open[idx])
            lots  = int(q)   #一張=lot size的股數 這裡設為1000

            # 以現金上限再約束一次，避免浮點誤差
            lots = min(lots, self._max_affordable_lots(price))

            if lots >= 1 and price > 0:
                """
                shares = 實際買的股數（張數 × 每張股數）。
                gross = 總成交金額（股數 × 單價）。
                fee = 手續費（買進手續費）。
                cash_out = 總現金流出 = 成交金額 + 手續費。
                """
                shares = lots * self.lot_size
                gross  = int(round(shares * price)) 
                fee    = int(round(gross * self.fee_buy))
                cash_out = gross + fee

                if cash_out <= self.cash + 1e-6:   #現金夠的情況下更新持股數量,現金,並更新交易紀錄
                    self.shares[idx] += shares
                    self.cash        -= cash_out
                    side, exec_shares, gross_cash, fees_tax = "BUY", shares, -gross, fee

        elif op == 1 and 0 <= idx < self.N and mask[1, idx, 0]:  #賣出全部股票情況
            # SELL_ALL: 只賣該 idx 一檔的全部
            shares = int(self.shares[idx])

            if shares > 0 and p_open[idx] > 0:
                price = float(p_open[idx])
                gross = int(round(shares * price))
                fee   = int(round(gross * self.fee_sell))
                tax   = int(round(gross * self.tax_sell))
                cash_in = gross - fee - tax
                self.shares[idx] = 0
                self.cash       += cash_in
                side, exec_shares, gross_cash, fees_tax = "SELL_ALL", -shares, gross, fee + tax

        # HOLD：其他情況不動

        # 估值 & 報酬（log-return）
        """
        Vprev = 上一期的total asset
        Vnew  = 新的total asset
        reward = 對數報酬 log(V_new/V_prev)
        """
        V_prev = float(self.portfolio_value)
        V_new  = float(self._mark_to_market(p_close))
        self.portfolio_value = V_new
        reward = float(np.log(max(V_new, 1e-12) / max(V_prev, 1e-12)))

        # 時間推進
        self._t += 1
        terminated = (self._t + 1 >= self.T)

        obs = self._make_obs(self._t) #新obs給下一個agent決策用

        # 準備下一步的遮罩（若已終局則為 HOLD-only；_build_action_mask 已處理）
        next_mask = self._build_action_mask(self._t)

        info = {
            "V": int(self.portfolio_value),
            "date": str(pd.Timestamp(self.dates[self._t]).date()),
            "side": side,
            "stock_id": (self.ids[idx] if 0 <= idx < self.N else None),
            "lots": (abs(exec_shares) // self.lot_size),
            "exec_shares": int(exec_shares),      # +買入股數 / -賣出股數（全賣）
            "gross_cash": int(gross_cash),        # 交易總額（未含費稅；買為負、賣為正）
            "fees_tax": int(fees_tax),            # 費用+稅
            "cash": int(round(self.cash)),
            "held": int((self.shares > 0).sum()),
            "action_mask_3d": next_mask,
        }
        return obs, reward, terminated, False, info   #回傳Step結果

    # endregion GymAPI
