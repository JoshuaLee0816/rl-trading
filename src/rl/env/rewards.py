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
        reward -= 0.00005

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

    return reward, baseline_return
# endregion Daily Return
