"""
ppo_test.py
用途：載入已訓練的 PPO Actor,對單一年度或多年度測試,輸出績效(Total Return、Max Drawdown)與圖表／交易紀錄。
支援策略：
- policy="argmax":以機率最大動作,並設置信心門檻(conf_threshold)不足則 HOLD
- policy="ev_greedy":Top-K 一階展望，用 Critic 的 V(s') 評分，選期望值最高動作
"""

import torch
import os
import sys
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 訓練中評測避免 block GUI
import matplotlib.pyplot as plt
from pathlib import Path

# === 專案路徑 ===
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
    conf_threshold=0.75           
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

    # === 繪圖 ===
    fig = None
    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df_perf.index, df_perf["value"], label="Agent Portfolio")
        plt.plot(df_baseline.index, df_baseline["baseline"], label="Baseline (0050)", linestyle="--")
        # === 加交易標記 ===
        if len(actions) > 0:   # 這裡直接用 actions
            df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
            df_trades["date"] = pd.to_datetime(df_trades["date"])

            # 買點 (Buy) → 綠色三角形
            buy_points = df_trades[df_trades["side"] == "BUY"]
            plt.scatter(buy_points["date"], buy_points["value"],
                        marker="^", color="green", s=80, label="Buy")

            # 賣點 (Sell) → 紅色倒三角
            sell_points = df_trades[df_trades["side"] == "SELL_ALL"]
            plt.scatter(sell_points["date"], sell_points["value"],
                        marker="v", color="red", s=80, label="Sell")

        plt.title(f"Portfolio Value Over Time ({tag})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    # === 交易紀錄 ===
    if save_trades:
        out_path = Path("src/rl/test/testing_output") / f"trades_{tag}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
        df_trades.to_csv(out_path, index=False)

    # 回傳邏輯統一，盡量不分支爆炸
    if return_fig and save_trades:
        return total_return, max_drawdown, df_perf, df_baseline, fig, actions
    if return_fig and not save_trades:
        return total_return, max_drawdown, df_perf, df_baseline, fig
    if not return_fig and save_trades:
        return total_return, max_drawdown, df_perf, df_baseline, actions
    return total_return, max_drawdown, df_perf, df_baseline


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

def run_test_suite(
    actor_path: Path,
    config_path: Path,
    years=(2020, 2021, 2022, 2023, 2024),
    plot=True,
    save_trades=False,
    verbose=True,
    policy="argmax", # 透傳策略參數，避免每年測試時手動改
    conf_threshold=0.75,
    overlay_plot=False # 是否在 suite 結束後做「多年份疊圖」
):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = {}
    overlay_fig = None # 用於 overlay 的參考，避免外部再收集

    for y in years:
        data_path = _resolve_test_path(ROOT, cfg, y)
        if not data_path.exists():
            if verbose:
                print(f"[WARN] Test data not found for {y}: {data_path}")
            continue

        if save_trades:
            tr, mdd, df_perf, df_base, fig, trades = run_test_once(
                actor_path=str(actor_path),
                data_path=str(data_path),
                config_path=str(config_path),
                plot=plot,
                save_trades=save_trades,
                tag=str(y),
                verbose=verbose,
                return_fig=True,
                policy=policy,                    
                conf_threshold=conf_threshold,   
            )
        else:
            tr, mdd, df_perf, df_base, fig = run_test_once(
                actor_path=str(actor_path),
                data_path=str(data_path),
                config_path=str(config_path),
                plot=plot,
                save_trades=save_trades,
                tag=str(y),
                verbose=verbose,
                return_fig=True,
                policy=policy,                  
                conf_threshold=conf_threshold,    
            )
            trades = None

    # 多年份疊圖
    if overlay_plot and len(results) > 0:
        overlay_fig = plt.figure(figsize=(10, 6))
        for y, r in results.items():
            dfp = r.get("df_perf")
            dfb = r.get("df_baseline")
            if dfp is not None and not dfp.empty:
                plt.plot(dfp.index, dfp["value"], label=f"Agent {y}")
            if dfb is not None and "baseline" in dfb and not dfb["baseline"].isna().all():
                plt.plot(dfb.index, dfb["baseline"], linestyle="--", label=f"0050 {y}")
        plt.title("Portfolio vs Baseline (Multi-Year Overlay)")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # 統一放在 results[-1]
        results["_overlay_fig"] = overlay_fig
        results[y] = {
            "total_return": tr,
            "max_drawdown": mdd,
            "fig": fig,
            "trades": None,  
            "df_perf": df_perf,
            "df_baseline": df_base,
        }

    return results

# === 獨立執行 ===
if __name__ == "__main__":
    # 取最後一次 run 的權重
    run_dirs = sorted((ROOT / "logs" / "runs").glob("run_*"))
    latest_run = run_dirs[-1]
    ACTOR_PATH = latest_run / "ppo_actor.pt"
    CONFIG_PATH = ROOT / "config.yaml"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    YEARS = (2020, 2021, 2022, 2023, 2024)

    print(f"[INFO] Using actor checkpoint: {ACTOR_PATH}")
    print(f"[INFO] Years: {YEARS}")

    # 預設做一次多年份疊圖
    results = run_test_suite(
        ACTOR_PATH,
        CONFIG_PATH,
        years=YEARS,
        plot=True,
        save_trades=True,
        verbose=True,
        policy="argmax",        
        conf_threshold=0.75,
        overlay_plot=True
    )
    
    for y, r in results.items():
        print(f"[{y}] TR={r['total_return']:.2%}, MDD={r['max_drawdown']:.2%}")
        
    plt.show()
