import torch
import torch.nn as nn

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

        self.register_buffer("pe", pe)                                          # buffer: 不更新參數，但會搬到正確 device ???

    def forward(self, x):  # x: [B, K, D]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class TemporalTransformerEncoder(nn.Module):
    """
    宣告一個模組處理每一檔的(特徵*時間窗)
    每檔股票共享的時間編碼器: (B, N, F, K) -> (B, N, D)
    """
    def __init__(self, F, d_model=48, n_heads=2, n_layers=1, dropout=0.1):

        super().__init__()

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

    def forward(self, x):  # x: [B, N, F, K]
        B, N, F, K = x.shape

        # 調整維度順序，把時間 K 放在中間 -> 只是為了符合transformer預期的維度順序 (PyToch 官方transformerencoderlayer要求)
        x = x.permute(0,1,3,2)    # [B,N,F,K] -> [B,N,K,F]

        # 線性投影，把 F -> d_model   (每天的特徵向量 -> 投影到 Transformer 的工作空間 D 維)
        x = self.proj(x)          # [B,N,K,D]

        # 加位置編碼
        x = self.pe(x)            # [B,N,K,D]

        # 丟進 Transformer encoder (在 K 維上做 self-attention)
        x = self.encoder(x)       # [B,N,K,D]

        """
        經過transformer後, 輸出是[B,N,K,D]
        每一檔都有K天(Lookback)的D維embedding
        但是Actor/Critic只吃"固定"長度的向量 不能帶著時間軸K
        所以必須有一個 聚合(aggregation) 把K個embedding合成1個embedding

        但要注意 我認為mean pooling是有可能lose時間序列的關係性 之後可以考慮換乘attention pooling 現在先用baseline處理
        """
        # 對 K 做mean pooling → 一個股票一個向量
        z = x.mean(dim=2)         # [B,N,D]

        # LayerNorm 清理數值 避免梯度爆炸或不穩
        return self.out_ln(z)     # [B,N,D]
