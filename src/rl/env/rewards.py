import torch

# region Daily Return
def daily_return(env, action, side, p_close, t):
    """
    Reward = Portfolio log return - Baseline log return - Transaction cost - Penalty
    -- Log Return : 代表真實持倉報酬的log
    -- Log baseline : 代表0050長期持有的指數
    -- Transaction cost: 每次sell_out跟Buy會扣出 " " 的成本, 避免過度交易
    -- Penalty : 如果持倉出現浮動虧損, 虧損越大penalty越大, 鼓勵agent學習指損避免長期虧損持有攤平
    """
    # 投資真實Log報酬
    V_prev = env.portfolio_value
    V_new = env._mark_to_market(p_close)
    env.portfolio_value = V_new

    # 更新 peak_value
    env.peak_value = torch.max(env.peak_value, V_new)

    portfolio_return = torch.log(
        torch.clamp(V_new, min=1e-12) / torch.clamp(V_prev, min=1e-12)
    )

    # 基準報酬
    baseline_return = torch.log(
        env.baseline_close[t + 1] / env.baseline_close[t]
    )

    reward = portfolio_return - baseline_return

    # Transaction Cost
    if side in ("BUY", "SELL_ALL"):
        reward *= 0.999

    # Penalty
    penalty = torch.tensor(0.0, device=env.device)
    for s, i in enumerate(env.slots):
        if i is not None and env.avg_costs[i] > 0:
            cur_price = env.prices_close[env._t, i]
            floating_ret = (cur_price - env.avg_costs[i]) / env.avg_costs[i]
            if floating_ret < 0:
                # 指數型懲罰
                penalty += (torch.exp(-5 * floating_ret) - 1) * 0.001
    reward -= penalty

    # === Penalty (MDD懲罰) ===
    dd_from_peak = (V_new - env.peak_value) / env.peak_value
    if dd_from_peak < -0.1:  # 超過 -10% 回撤
        mdd_penalty = dd_from_peak * 0.8  # 懲罰力度可調
        reward += mdd_penalty

    return reward, {
        "baseline_return": float(baseline_return.item()),
        "mdd": float(dd_from_peak.item())
    }
# endregion Daily Return

# region Strong_signal_return
def strong_signal_return(env, action, side, p_close, t):
    """
    Reward = 強化梯度版日報酬
    目標：提升訓練初期學習速度與穩定度

    結構：
        r = α * (portfolio_return - β * baseline_return)
            - λ1 * transaction_cost
            - λ2 * floating_loss_penalty
            - λ3 * drawdown_penalty
    """

    # === Portfolio Value 更新 ===
    V_prev = env.portfolio_value
    V_new = env._mark_to_market(p_close)
    env.portfolio_value = V_new

    env.peak_value = torch.max(env.peak_value, V_new)

    # === 核心報酬 ===
    portfolio_return = torch.log(
        torch.clamp(V_new, min=1e-12) / torch.clamp(V_prev, min=1e-12)
    )
    baseline_return = torch.log(env.baseline_close[t + 1] / env.baseline_close[t])

    # === 放大回報信號，降低baseline權重 ===
    alpha = 8.0   # 放大倍率 (強化梯度)
    beta = 0.3     # baseline 權重
    reward = alpha * (portfolio_return - beta * baseline_return)

    # === Transaction cost ===
    if side in ("BUY", "SELL_ALL"):
        reward -= 0.02  # 避免過度交易但不削弱信號

    # === 平滑浮虧 penalty ===
    penalty = torch.tensor(0.0, device=env.device)
    for i in env.slots:
        if i is not None and env.avg_costs[i] > 0:
            cur_price = env.prices_close[env._t, i]
            floating_ret = (cur_price - env.avg_costs[i]) / env.avg_costs[i]
            if floating_ret < 0:
                # 改用線性平滑懲罰
                penalty += (-floating_ret) * 0.02
    reward -= penalty

    # === Drawdown penalty ===
    dd_from_peak = (V_new - env.peak_value) / env.peak_value
    if dd_from_peak < -0.15:
        reward += dd_from_peak * 0.1  # 線性懲罰，避免爆梯度

    # === Clip reward to keep gradient stable ===
    reward = torch.clamp(reward, -1.0, 1.0)

    # 觀察數量級用
    """
    print(f"[DEBUG] portfolio_return={portfolio_return.mean():.6f}, "
          f"drawdown={dd_from_peak:.4f}, penalty={penalty.item():.4f}, "
          f"reward={reward.mean():.4f}")
    """
    
    return reward, {
        "baseline_return": float(baseline_return.item()),
        "mdd": float(dd_from_peak.item()),
        "penalty": float(penalty.item())
    }
# endregion 強化訓練績效 Reward

# region Aggressive_signal_return_v2 (vector-safe)
def aggressive_signal_return(env, action, side, p_close, t):
    """
    Reward = 積極獲利導向版 (支援多環境)
    修正版：移除所有會造成 ambiguous truth value 的 if
    """

    V_prev = env.portfolio_value
    V_new = env._mark_to_market(p_close)
    env.portfolio_value = V_new
    env.peak_value = torch.max(env.peak_value, V_new)

    # === Return ===
    portfolio_return = torch.log(torch.clamp(V_new / V_prev, min=1e-12))
    baseline_return = torch.log(env.baseline_close[t + 1] / env.baseline_close[t])

    # === 改 1: 強化 outperform，正報酬平方放大 ===
    raw = portfolio_return - 0.3 * baseline_return
    reward = torch.where(raw > 0, 1.5 * raw * (1 + 3 * raw), raw)   # ✅ 向量化

    # === 改 2: 降低固定交易成本 ===
    if isinstance(side, (list, tuple)):
        # 多環境：批次處理
        side_tensor = torch.tensor([1.0 if s in ("BUY", "SELL_ALL") else 0.0 for s in side], device=env.device)
        reward -= 0.01 * side_tensor
    else:
        # 單環境
        if side in ("BUY", "SELL_ALL"):
            reward -= 0.01

    # === 改 3: 漸進式浮虧懲罰 (向量化)
    penalty = torch.zeros_like(reward)
    for i in env.slots:
        if i is not None and env.avg_costs[i] > 0:
            cur_price = env.prices_close[env._t, i]
            floating_ret = (cur_price - env.avg_costs[i]) / env.avg_costs[i]
            penalty += torch.where(floating_ret < 0, -floating_ret * 0.015, torch.zeros_like(floating_ret))
    reward -= penalty

    # === 改 4: Drawdown penalty (向量化)
    dd_from_peak = (V_new - env.peak_value) / env.peak_value
    drawdown_penalty = torch.where(dd_from_peak < -0.2, dd_from_peak * 0.08, torch.zeros_like(dd_from_peak))
    reward += drawdown_penalty

    # === 改 5: Idle penalty (向量化)
    if isinstance(action, torch.Tensor):
        hold_mask = (action == 0).float()
        reward -= 0.001 * hold_mask
    else:
        if action == 0:
            reward -= 0.001

    # === Clamp 防梯度爆炸 ===
    reward = torch.clamp(reward, -1.0, 1.0)

    return reward, {
        "baseline_return": float(torch.mean(baseline_return).item()),
        "mdd": float(torch.mean(dd_from_peak).item()),
        "penalty": float(torch.mean(penalty).item())
    }
# endregion



def get_reward_fn(mode: str):
    if mode == "daily_return":
        return daily_return
    
    elif mode == "strong_signal_return":
        return strong_signal_return

    elif mode == "aggressive_signal_return":
        return aggressive_signal_return
    
    else:
        raise ValueError(f"Unknown reward mode: {mode}")

# endregion 取得def
