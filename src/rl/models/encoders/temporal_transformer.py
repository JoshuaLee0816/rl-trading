import torch
import torch.nn as nn
import time

class PositionalEncoding(nn.Module):
    """
    生成位置編碼
    目的:?????
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                                      # 這個矩陣存每個時間位置的編碼向量
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)        # 建立位置向量 unsqueeze for 後續傳播
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0))/d_model))

        """
        偶數維度放sin值
        奇數維度放cos值
        """
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))                             # [1, max_len, D]  

    def forward(self, x):  # x: [B, K, D]
        L = x.size(1)
        return x + self.pe[:, :L, :]   # [1,L,D] + [B,L,D]

class TemporalTransformerEncoder(nn.Module):
    """
    宣告一個模組處理每一檔的(特徵*時間窗)
    每檔股票共享的時間編碼器: (B, N, F, K) -> (B, N, D)
    """
    def __init__(self, F, d_model=48, n_heads=2, n_layers=1, dropout=0.1,k_window=20):

        super().__init__()
        self.F = F
        self.k_window = k_window

        # 1) 先把每一天的 F 個特徵投影到 d_model 維 -> 得到一個每日embedding
        self.proj = nn.Linear(F, d_model)

        # 2) Transformer encoder layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 3) 位置編碼 + 輸出正規化
        self.pe = PositionalEncoding(d_model)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, K, N, F]
        # 確保輸入有 batch 維度
        if x.dim() == 3:   # [K, N, F]
            x = x.unsqueeze(0)  # [1, K, N, F]

        B, K, N, F = x.shape

        # reshape → 把 B,N 合併，保留時間 K 和特徵 F
        x = x.reshape(B * N, K, F)      # [B*N, K, F]

        # 線性投影 (F → d_model)
        x = self.proj(x)                # [B*N, K, D]

        # 加位置編碼
        x = self.pe(x)                  # [B*N, K, D]

        # 丟進 Transformer
        x = self.encoder(x)             # [B*N, K, D]

        # mean pooling over K
        z = x.mean(dim=1)               # [B*N, D]

        # reshape 回 [B, N, D]
        z = z.view(B, N, -1)            # [B, N, D]

        # LayerNorm
        out = self.out_ln(z)            # [B, N, D]

        return out
