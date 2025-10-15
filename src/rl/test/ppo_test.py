"""
ppo_test.py
用途：載入已訓練的 PPO Actor,對單一年度或多年度測試,輸出績效(Total Return、Max Drawdown)與圖表／交易紀錄。
支援策略：
- policy="argmax":以機率最大動作,並設置信心門檻(conf_threshold)不足則 HOLD
- policy="ev_greedy":Top-K 一階展望，用 Critic 的 V(s') 評分，選期望值最高動作
"""

import os
import sys
import numpy as np

import matplotlib
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")  # 訓練中評測避免 block GUI
from pathlib import Path

import matplotlib.pyplot as plt

# region 專案路徑
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]
ROOT = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl.env.StockTradingEnv import StockTradingEnv
from rl.models.ppo_agent import PPOAgent


# === 環境快照/還原（只在測試端用，一階 lookahead） ===
def _snapshot_env(env):
    return {
        "_t": env._t,
        "cash": env.cash.clone(),
        "shares": env.shares.clone(),
        "avg_costs": env.avg_costs.clone(),
        "slots": list(env.slots),
        "portfolio_value": env.portfolio_value.clone(),
    }

def _restore_env(env, snap):
    env._t = snap["_t"]
    env.cash = snap["cash"]
    env.shares = snap["shares"]
    env.avg_costs = snap["avg_costs"]
    env.slots = list(snap["slots"])
    env.portfolio_value = snap["portfolio_value"]

# region run_test_once
def run_test_once(
    actor_path,
    data_path,
    config_path,
    plot=True,
    save_trades=False,
    tag="2020",
    verbose=True,
    return_fig=False,
    policy="argmax",              
    conf_threshold=0.75,
    initial_cash=None,            
):        
    # === 載入 config.yaml ===
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    env_cfg = full_cfg["environment"]
    feature_cols = full_cfg["data"]["features"]

    # === 載入測試資料 ===
    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])
    keep_cols = ["date", "stock_id"] + feature_cols
    df = df[keep_cols]
    ids = sorted(df["stock_id"].unique())

    # override initial cash if provided
    if initial_cash is not None:
        env_cfg["initial_cash"] = initial_cash
        
    # === 初始化環境（CPU） ===
    env = StockTradingEnv(
        df=df,
        stock_ids=ids,
        lookback=env_cfg["lookback"],
        initial_cash=env_cfg["initial_cash"],
        max_holdings=env_cfg["max_holdings"],
        qmax_per_trade=env_cfg["qmax_per_trade"],
        device="cpu",
    )

    obs, info = env.reset()
    obs_dim = env.obs_dim

    # === 建立 PPO Agent ===
    agent = PPOAgent(obs_dim, len(ids), env.QMAX, full_cfg)

    # === 載入已訓練好的 actor 參數 ===
    if full_cfg["training"].get("load_checkpoint", True) and actor_path is not None and os.path.exists(actor_path):
        try:
            ckpt = torch.load(actor_path, map_location=agent.device)
            if isinstance(ckpt, dict) and "actor" in ckpt:
                agent.actor.load_state_dict(ckpt["actor"])
            else:
                agent.actor.load_state_dict(ckpt)
            agent.actor.eval()
            if verbose:
                print(f"[INFO] Loaded checkpoint from {actor_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Using random init.")
    else:
        if verbose:
            pass

    # === 測試 loop（支援 argmax / ev_greedy） ===
    dates, values, actions = [], [], []
    trade_records = [] #存交易紀錄

    terminated = False

    # 小工具：從 mask 拿到所有合法動作（flat 索引與對應 tuple）
    def _legal_actions_from_mask(mask_flat_1d):
        idxs = torch.nonzero(mask_flat_1d, as_tuple=False).view(-1).tolist()
        return idxs, [agent.flat_to_tuple(i) for i in idxs]

    while not terminated:
        with torch.no_grad():
            obs_t = agent.obs_to_tensor(obs).unsqueeze(0)  # [1,obs_dim]
            mask_flat = agent.flatten_mask(info["action_mask_3d"])  # [A]
            logits = agent.actor(obs_t)
            masked_logits = logits.masked_fill(~mask_flat.unsqueeze(0), -1e9)

            if policy == "argmax":
                # === softmax 取得機率分布 ===
                probs = torch.softmax(masked_logits, dim=-1)
                max_prob, a_flat = torch.max(probs, dim=-1)

                # === 信心閾值判斷 ===
                if max_prob.item() >= conf_threshold:
                    action_tuple = agent.flat_to_tuple(int(a_flat.item()))
                else:
                    # 信心不足 : 選 HOLD
                    action_tuple = (2, 0, 0)   # MultiDiscrete([op, idx, q]) 裡 2=HOLD

            elif policy == "sample":
                # === 以機率抽樣動作（模擬訓練行為） ===
                probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
                dist = torch.distributions.Categorical(probs)
                a_flat = dist.sample()
                action_tuple = agent.flat_to_tuple(int(a_flat.item()))

            elif policy == "ev_greedy":
                # === Top-K EV-greedy：僅模擬前 K 個動作，依 Actor 機率挑選 ===
                TOPK = 10  # 只模擬前 10 個動作
                legal_flat, _ = _legal_actions_from_mask(mask_flat)

                # 若沒有合法動作就直接 HOLD
                if len(legal_flat) == 0:
                    action_tuple = (2, 0, 0)
                else:
                    # 先根據 actor 機率分布選出前 K 個動作
                    probs = torch.softmax(masked_logits, dim=-1).squeeze(0)  # [A]
                    k = min(TOPK, int(mask_flat.sum().item()))
                    topk = torch.topk(probs, k=k)            # k 上限以實際可行動作數為準
                    top_actions = [agent.flat_to_tuple(int(i)) for i in topk.indices.tolist()]

                    # 對這些動作各別模擬一步，用 critic 的 V(s') 打分
                    best_v = -1e30
                    best_action = (2, 0, 0)  # 預設 HOLD
                    for a_tuple in top_actions:
                        snap = _snapshot_env(env)
                        try:
                            next_obs, _, sim_term, sim_trunc, next_info = env.step(a_tuple)
                            next_obs_t = agent.obs_to_tensor(next_obs).unsqueeze(0)
                            next_obs_t = (next_obs_t - next_obs_t.mean(dim=1, keepdim=True)) / (
                                next_obs_t.std(dim=1, keepdim=True) + 1e-8
                            )
                            v = agent.critic(next_obs_t).item()
                        except Exception as e:
                            print(f"[ERROR][EV-greedy] step({a_tuple}) failed: {e}")
                            v = -1e30
                        _restore_env(env, snap)
                        if v > best_v:
                            best_v = v
                            best_action = a_tuple

                    # 最後選出期望值最高的動作
                    action_tuple = best_action

            else:
                raise ValueError(f"Unknown policy: {policy}")


        obs, reward, terminated, _, info = env.step(action_tuple)

        dates.append(info["date"])
        values.append(info["V"])
        actions.append((info["date"], info["side"], info["stock_id"], info["lots"], info["cash"], info["V"]))

        # [新增] 每筆交易紀錄（只記 BUY/SELL，不記 HOLD）
        if info["side"] in ["BUY", "SELL_ALL"]:
            trade_records.append({
                "date": info["date"],
                "action": info["side"],
                "stock_id": info["stock_id"],
                "lots": info["lots"],
                "price": round(info["price"], 2) if "price" in info else None,
                "cash_after": round(info["cash"], 2),
                "portfolio_value": round(info["V"], 2),
                "reward": round(reward, 4),
            })

    # === 績效分析 ===
    df_perf = pd.DataFrame({"date": dates, "value": values})
    df_perf["date"] = pd.to_datetime(df_perf["date"])
    df_perf.set_index("date", inplace=True)

    total_return = df_perf["value"].iloc[-1] / df_perf["value"].iloc[0] - 1
    roll_max = df_perf["value"].cummax()
    drawdown = df_perf["value"] / roll_max - 1
    max_drawdown = drawdown.min()

    # === baseline（0050） ===
    try:
        baseline_value = (env.baseline_close / env.baseline_close[env.K]) * env.initial_cash
        df_baseline = pd.DataFrame({"date": env.dates[env.K:], "baseline": baseline_value[env.K:].cpu().numpy()})
        df_baseline["date"] = pd.to_datetime(df_baseline["date"])
        df_baseline.set_index("date", inplace=True)
    except Exception as e:
        if verbose:
            print(f"[WARN] Baseline unavailable: {e}")
        df_baseline = pd.DataFrame(index=df_perf.index, data={"baseline": float("nan")})    

    # 繪圖
    fig = None
    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df_perf.index, df_perf["value"], label="Agent Portfolio")
        plt.plot(df_baseline.index, df_baseline["baseline"], label="Baseline (0050)", linestyle="--")
        # 交易標記
        if len(actions) > 0:   # 這裡直接用 actions
            df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
            df_trades["date"] = pd.to_datetime(df_trades["date"])

            # Buy → 綠色三角形
            buy_points = df_trades[df_trades["side"] == "BUY"]
            plt.scatter(buy_points["date"], buy_points["value"],
                        marker="^", color="green", s=80, label="Buy")

            # Sell → 紅色倒三角
            sell_points = df_trades[df_trades["side"] == "SELL_ALL"]
            plt.scatter(sell_points["date"], sell_points["value"],
                        marker="v", color="red", s=80, label="Sell")

        plt.title(f"Portfolio Value Over Time ({tag})")

        # 在圖上右上角顯示交易次數與報酬率
        sell_count = sum(
            1 for r in trade_records
            if r["action"] == "SELL_ALL" and r.get("lots", 0) > 0 and r.get("price") is not None
        )
        return_pct = total_return * 100
        text_str = f"Trades: {sell_count}\nReturn: {return_pct:+.2f}%"
        plt.text(
            0.98, 0.02, text_str,
            transform=plt.gca().transAxes,
            fontsize=11,
            color="black",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.6)
        )

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    # === 交易紀錄 ===
    if save_trades:

        # 加上 episode 編號避免覆蓋 ===
        out_path = Path("src/rl/test/testing_output") / f"trades_{tag}.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades = pd.DataFrame(trade_records)
        # 新增報酬率欄位（相對於第一筆的 Portfolio Value）
        if len(df_trades) > 0:
            first_value = df_trades["portfolio_value"].iloc[0]
            df_trades["return(%)"] = (df_trades["portfolio_value"] / first_value - 1) * 100

        df_trades["trade_profit"] = None
        df_trades["trade_return(%)"] = None

        buy_price, buy_lots = None, None
        for i, row in df_trades.iterrows():
            if row["action"] == "BUY":
                buy_price = row["price"]
                buy_lots = row["lots"]
            elif row["action"] == "SELL_ALL" and buy_price is not None:
                sell_price = row["price"]
                profit = (sell_price - buy_price) * buy_lots * 1000  # 單位元
                trade_return = (sell_price / buy_price - 1) * 100
                df_trades.loc[i, "trade_profit"] = round(profit, 2)
                df_trades.loc[i, "trade_return(%)"] = round(trade_return, 2)
                # 清空 buy 狀態，避免跨股票混算
                buy_price, buy_lots = None, None

        df_trades.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 交易紀錄已儲存：{out_path}")

    # 回傳邏輯統一，盡量不分支爆炸
    if return_fig and save_trades:
        sell_count = sum(
            1 for r in trade_records
            if r["action"] == "SELL_ALL" and r.get("lots", 0) > 0 and r.get("price") is not None
        )
        return total_return, max_drawdown, df_perf, df_baseline, fig, actions, sell_count
    if return_fig and not save_trades:
        return total_return, max_drawdown, df_perf, df_baseline, fig
    if not return_fig and save_trades:
        return total_return, max_drawdown, df_perf, df_baseline, actions
    return total_return, max_drawdown, df_perf, df_baseline


# region run_test_random_start

def run_test_random_start(
    actor_path,
    config_path,
    plot=True,
    save_trades=False,
    tag=None,
    verbose=True,
    return_fig=False,
    policy="argmax",
    conf_threshold=0.75,
    random_seed=None,
    n_runs=5,
):
    """
    從 2020~2024 整合測試檔隨機抽取多段進行完整測試：
    - 固定長度 252 交易日（一年）
    - 起始時間隨機（需 ≥ lookback）
    - 初始資金隨機 (10萬~50萬)
    - 每段呼叫 run_test_once()，畫出完整 Portfolio 曲線圖（含 baseline、買賣點）
    - 回傳平均績效與所有圖形
    """

    import random
    import matplotlib.pyplot as plt

    # === 載入 config.yaml ===
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    env_cfg = full_cfg["environment"]
    feature_cols = full_cfg["data"]["features"]
    lookback = env_cfg["lookback"]

    # === 載入完整測試資料 ===
    if "test_file" in full_cfg["data"]:
        data_path = Path(full_cfg["data"]["test_file"])
    else:
        data_path = Path(full_cfg["data"]["file"])

    ROOT = Path(config_path).resolve().parent
    data_path = ROOT / "data" / "processed" / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"測試資料不存在：{data_path}")

    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"[INFO] 載入測試資料成功：{data_path}")

    keep_cols = ["date", "stock_id"] + feature_cols
    df = df[keep_cols]
    df = df.sort_values("date").reset_index(drop=True)
    ids = sorted(df["stock_id"].unique())

    # === 日期範圍檢查 ===
    all_dates = np.sort(df["date"].unique())
    total_days = len(all_dates)
    if total_days <= lookback + 252:
        raise ValueError("資料長度不足，無法抽出一年測試區間")

    # === 結果記錄 ===
    results = []
    figs = []
    all_sell_counts = []

    rng = np.random.default_rng(random_seed)
    for run_id in range(1, n_runs + 1):
        # 隨機一年區段
        start_idx = rng.integers(lookback, total_days - 252)
        end_idx = start_idx + 252
        start_date, end_date = all_dates[start_idx], all_dates[min(end_idx, total_days - 1)]
        df_sub = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        # 隨機初始資金
        init_cash = float(rng.integers(100_000, 500_001))
        local_tag = f"{tag or 'random'}_{run_id}_{str(start_date)[:10]}"

        print(f"[Random Test {run_id}/{n_runs}] 區間: {start_date} ~ {end_date} | 初始資金: {init_cash:,.0f}")

        # === 暫存子資料 ===
        tmp_path = Path("src/rl/test/temp_random.parquet")
        df_sub.to_parquet(tmp_path, index=False)

        try:
            total_return, max_drawdown, df_perf, df_base, fig, actions, sell_count = run_test_once(
                actor_path=str(actor_path),
                data_path=str(tmp_path),
                config_path=config_path,
                plot=False,            # 先不畫，下面自己畫真實金額版
                save_trades=save_trades,
                tag=local_tag,
                verbose=False,
                return_fig=True,
                policy=policy,
                conf_threshold=conf_threshold,
                initial_cash=init_cash, 
            )

            # === 🎨 真實金額繪圖 ===
            fig = plt.figure(figsize=(10, 6))
            plt.plot(df_perf.index, df_perf["value"], label=f"Agent Portfolio (Init={init_cash:,.0f})")

            # baseline 真實比例
            try:
                baseline_value = (df_base["baseline"] / df_base["baseline"].iloc[0]) * init_cash
                plt.plot(df_base.index, baseline_value, label="Baseline (0050)", linestyle="--")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Baseline unavailable in random test: {e}")

            # 交易標記
            if len(actions) > 0:
                df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
                df_trades["date"] = pd.to_datetime(df_trades["date"])
                buy_points = df_trades[df_trades["side"] == "BUY"]
                plt.scatter(buy_points["date"], buy_points["value"], marker="^", color="green", s=80, label="Buy")
                sell_points = df_trades[df_trades["side"] == "SELL_ALL"]
                plt.scatter(sell_points["date"], sell_points["value"], marker="v", color="red", s=80, label="Sell")

            # 標題 + 資訊文字
            plt.title(f"Portfolio Value Over Time ({local_tag})")
            return_pct = total_return * 100
            text_str = f"Init: {init_cash:,.0f}\nReturn: {return_pct:+.2f}%\nTrades: {sell_count}"
            plt.text(
                0.98, 0.02, text_str,
                transform=plt.gca().transAxes,
                fontsize=11,
                color="black",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.6)
            )
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value (NTD)")
            plt.legend()
            plt.grid(True)

            # === 統計記錄 ===
            results.append((total_return, max_drawdown))
            figs.append(fig)
            all_sell_counts.append(sell_count)

        except Exception as e:
            print(f"[WARN] Random Test {run_id} 失敗：{e}")
            continue

    # === 匯總結果 ===
    mean_return = np.mean([r[0] for r in results]) if results else 0.0
    mean_mdd = np.mean([r[1] for r in results]) if results else 0.0
    mean_trades = np.mean(all_sell_counts) if all_sell_counts else 0.0

    # === 回傳 ===
    return {
        "total_return": mean_return,
        "max_drawdown": mean_mdd,
        "sell_count": mean_trades,
        "figs": figs,  # ✅ 多張 Portfolio 圖 (每段測試一張)
    }




def _resolve_test_path(root: Path, cfg: dict, year: int) -> Path:
    """
    支援兩種設定：
    1) cfg["data"]["test_files"] = {"2020":"...", ...}
    2) fallback pattern: data/processed/test_{year}.parquet 或 .csv
    """
    data_cfg = cfg.get("data", {})
    test_files = data_cfg.get("test_files")
    if isinstance(test_files, dict) and str(year) in test_files:
        return root / "data" / "processed" / test_files[str(year)]
    # fallback pattern
    p_parquet = root / "data" / "processed" / f"test_{year}.parquet"
    p_csv     = root / "data" / "processed" / f"test_{year}.csv"
    return p_parquet if p_parquet.exists() else p_csv
