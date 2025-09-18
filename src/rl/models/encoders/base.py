import torch.nn as nn

class BaseEncoder(nn.Module):
    """
    輸入: x 形狀 (B, N, F, K)
    輸出: z 形狀 (B, N, D)


    輸入:x 的形狀是 (B, N, F, K)

    B: batch size(一次幾個樣本）

    N: 股票數(universe 裡的檔數，例如 20、300)

    F: 每檔股票的特徵數(features,例如 7 條:open, close, volume…)

    K: lookback 視窗長度（例如最近 20 天）

    輸出:z 的形狀是 (B, N, D)

    D: encoder 最後輸出的維度（壓縮後的 embedding 大小，例如 32 或 48)

    保持 N 維度不變 → 代表「每檔都對應一個向量表示」。
    """
    def forward(self, x):
        raise NotImplementedError
