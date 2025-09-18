import torch.nn as nn

class BaseEncoder(nn.Module):
    """
    輸入: x 形狀 (B, N, F, K)
    輸出: z 形狀 (B, N, D)

    B: batch size (一次幾個樣本)
    N: 股票數 (universe 裡的檔數，例如 20、300)
    F: 每檔股票的特徵數 (features，例如 7 條: open, close, volume…)
    K: lookback 視窗長度 (例如最近 20 天)

    輸出: z 的形狀 (B, N, D)
    D: encoder 最後輸出的維度 (壓縮後的 embedding 大小，例如 32 或 48)

    保持 N 維度不變 → 每檔股票對應一個向量表示。
    """
    def forward(self, x):
        raise NotImplementedError


# --- Identity Encoder（不做任何壓縮，直接攤平） ---
class IdentityEncoder(BaseEncoder):
    def __init__(self, F, k_window):
        super().__init__()
        self.F = F
        self.k_window = k_window

    def forward(self, x):  # x: [B,N,F,K]
        B, N, F, K = x.shape
        return x.reshape(B, N, F * K)  # [B, N, F*K]


# --- 函式 ---
def build_encoder(cfg, F):
    enc_type = cfg.get("type", "identity")

    if enc_type == "identity":
        return IdentityEncoder(F, cfg["params"].get("k_window", 20))

    elif enc_type == "temporal_transformer":
        from .temporal_transformer import TemporalTransformerEncoder
        return TemporalTransformerEncoder(
            F=F,
            d_model=cfg["params"].get("d_model", 48),
            n_heads=cfg["params"].get("n_heads", 2),
            n_layers=cfg["params"].get("n_layers", 1),
            dropout=cfg["params"].get("dropout", 0.1),
            k_window=cfg["params"].get("k_window", 20),
        )

    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
