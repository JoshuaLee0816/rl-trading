from pathlib import Path
import pandas as pd

class RunLogger:
    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.eq_rows = []
        self.trade_rows = []

    def log_step(self, ep: int, step_i: int, reward: float, info: dict):
        # equity curve 每步都記
        self.eq_rows.append({
            "episode": ep,
            "step": step_i,
            "date": info.get("date"),
            "V": info.get("V"),
            "cash": info.get("cash"),
            "reward": float(reward),
        })
        # 有成交才記交易
        if info.get("exec_shares", 0) != 0:
            self.trade_rows.append({
                "episode": ep,
                "date": info.get("date"),
                "stock_id": info.get("stock_id"),
                "side": info.get("side"),
                "qty": int(abs(info.get("exec_shares", 0))),
                "price_open": info.get("price_open"),
                "price_close": info.get("price_close"),
                "fees_tax": info.get("fees_tax"),
                "V_after": info.get("V"),
                "cash_after": info.get("cash"),
            })

    def save_csv(self, tag: str):
        # 存成兩張表：一張 equity、一張 trades
        eq = pd.DataFrame(self.eq_rows)
        tr = pd.DataFrame(self.trade_rows)
        eq.to_csv(self.outdir / f"equity_{tag}.csv", index=False)
        tr.to_csv(self.outdir / f"trades_{tag}.csv", index=False)
        return (self.outdir / f"equity_{tag}.csv", self.outdir / f"trades_{tag}.csv")
