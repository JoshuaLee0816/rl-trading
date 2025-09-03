from pathlib import Path
import pandas as pd

class RunLogger:
    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.eq_rows = []
        self.trade_rows = []
        self.trade_count = 0

    def log_step(self, ep: int, info: dict):
        '''
        # equity curve 每步都記
        self.eq_rows.append({
            "episode": ep,
            "step": step_i,
            "date": info.get("date"),
            "V": info.get("V"),
            "cash": info.get("cash"),
            "reward": float(reward),
        })
        '''
        # 有成交才記交易
        if info.get("exec_shares", 0) != 0:
            if self.trade_count % 10 == 0:   #這邊調整要存多少比例的訓練資料
                row = {
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
                }
                pd.DataFrame([row]).to_csv(
                    self.outdir / "trades_all.csv",
                    mode="a", header=not (self.outdir / "trades_all.csv").exists(), index=False
            )
        self.trade_count += 1

    def save_csv(self, tag: str):
        
        #eq = pd.DataFrame(self.eq_rows)
        #eq.to_csv(self.outdir / f"equity_{tag}.csv", index=False)

        tr = pd.DataFrame(self.trade_rows)
        tr.to_csv(self.outdir / f"trades_{tag}.csv", index=False)
        return (self.outdir / f"trades_{tag}.csv")
