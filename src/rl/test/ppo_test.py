import torch
import os
import sys
import yaml
import pandas as pd
import numpy as np
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

# === 新增：環境快照/還原（只在測試端用，一階 lookahead） ===
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


def run_test_once(actor_path, data_path, config_path,
                  plot=True, save_trades=False, tag="2020", verbose=True,
                  return_fig=False,
                  policy="argmax",              # <<< 新增：策略切換（argmax / ev_greedy）
                  conf_threshold=0.75):         # <<< 新增：argmax 信心門檻
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
            #print("[INFO] No checkpoint found or load_checkpoint=False. Using random init.")
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
                    # 信心不足 → 選 HOLD
                    action_tuple = (2, 0, 0)   # 假設 MultiDiscrete([op, idx, q]) 裡 2=HOLD

            elif policy == "ev_greedy":
                # 一階展望：對每個合法動作暫時 step 一次，用 Critic 的 V(s') 打分
                legal_flat, legal_tuples = _legal_actions_from_mask(mask_flat)
                best_v = -1e30
                best_action = (2, 0, 0)  # 預設 HOLD（通常合法）
                for a_flat_idx, a_tuple in zip(legal_flat, legal_tuples):
                    snap = _snapshot_env(env)
                    # 模擬一步
                    try:
                        next_obs, _, terminated, truncated, next_info = env.step(a_tuple)
                    except Exception as e:
                        print(f"[ERROR][EV-greedy] step({a_tuple}) failed: {e}")
                        _restore_env(env, snap)
                        continue
                    # 與 select_action 對齊：丟入 critic 前做每筆標準化
                    next_obs_t = agent.obs_to_tensor(next_obs).unsqueeze(0)
                    next_obs_t = (next_obs_t - next_obs_t.mean(dim=1, keepdim=True)) / (next_obs_t.std(dim=1, keepdim=True) + 1e-8)
                    v = agent.critic(next_obs_t).item()
                    # 還原環境
                    _restore_env(env, snap)
                    if v > best_v:
                        best_v = v
                        best_action = a_tuple
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

    """
    if verbose:
        print(f"[TEST-{tag}] Total Return: {total_return:.2%}, Max Drawdown: {max_drawdown:.2%}")
    """

    # === baseline（0050） ===
    baseline_value = (env.baseline_close / env.baseline_close[env.K]) * env.initial_cash
    df_baseline = pd.DataFrame({"date": env.dates[env.K:], "baseline": baseline_value[env.K:].cpu().numpy()})
    df_baseline["date"] = pd.to_datetime(df_baseline["date"])
    df_baseline.set_index("date", inplace=True)

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
        # 不呼叫 plt.show()；由訓練端或主程式自行處理

    # === 交易紀錄 ===
    if save_trades:
        out_path = Path("src/rl/test/testing_output") / f"trades_{tag}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
        df_trades.to_csv(out_path, index=False)

    # <<< 修改：在 save_trades=True 時，多回傳 trades
    if return_fig:
        if save_trades:
            return total_return, max_drawdown, df_perf, df_baseline, fig, actions   # <<< 修改
        else:
            return total_return, max_drawdown, df_perf, df_baseline, fig
    else:
        if save_trades:
            return total_return, max_drawdown, df_perf, df_baseline, actions        # <<< 修改
        else:
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

def run_test_suite(actor_path: Path, config_path: Path, years=(2020, 2021, 2022, 2023, 2024),
                   plot=True, save_trades=False, verbose=True):
    """
    依序跑多個年份測試；回傳 dict:
      results[year] = {
          "total_return": float,
          "max_drawdown": float,
          "fig": matplotlib.figure.Figure,
          "trades": list[dict] | None
      }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = {}
    for y in years:
        data_path = _resolve_test_path(ROOT, cfg, y)
        if not data_path.exists():
            if verbose:
                print(f"[WARN] Test data not found for {y}: {data_path}")
            continue

        if save_trades:
            # <<< 修改：接收 trades
            tr, mdd, _, _, fig, trades = run_test_once(
                actor_path=str(actor_path),
                data_path=str(data_path),
                config_path=str(config_path),
                plot=plot,
                save_trades=save_trades,
                tag=str(y),
                verbose=verbose,
                return_fig=True,
            )
            results[y] = {
                "total_return": tr,
                "max_drawdown": mdd,
                "fig": fig,
                "trades": trades,   # <<< 修改
            }
        else:
            tr, mdd, _, _, fig = run_test_once(
                actor_path=str(actor_path),
                data_path=str(data_path),
                config_path=str(config_path),
                plot=plot,
                save_trades=save_trades,
                tag=str(y),
                verbose=verbose,
                return_fig=True,
            )
            results[y] = {
                "total_return": tr,
                "max_drawdown": mdd,
                "fig": fig,
                "trades": None,    # <<< 修改：保持欄位一致
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

    results = run_test_suite(ACTOR_PATH, CONFIG_PATH, years=YEARS, plot=True, save_trades=True, verbose=True)
    for y, r in results.items():
        print(f"[{y}] TR={r['total_return']:.2%}, MDD={r['max_drawdown']:.2%}")
    plt.show()
