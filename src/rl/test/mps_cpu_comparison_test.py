import torch, time

# === 模擬你的專案維度 ===
N, F, K = 300, 11, 15
L = N * K   # sequence length
d = 64      # embedding dim (常見 transformer hidden size)

print(f"Benchmarking Attention: L={L}, d={d}")

# 建立隨機 tensor
q = torch.randn(1, L, d)
k = torch.randn(1, L, d)
v = torch.randn(1, L, d)

def run_attention(device):
    q_d, k_d, v_d = q.to(device), k.to(device), v.to(device)
    torch.mps.synchronize() if device.type == "mps" else None

    start = time.time()
    attn = (q_d @ k_d.transpose(-2, -1)) @ v_d
    if device.type == "mps":
        torch.mps.synchronize()
    return time.time() - start

# === CPU 測試 ===
cpu_time = run_attention(torch.device("cpu"))
print(f"CPU: {cpu_time:.4f} 秒")

# === MPS 測試（如果可用） ===
if torch.backends.mps.is_available():
    mps_time = run_attention(torch.device("mps"))
    print(f"MPS: {mps_time:.4f} 秒")
else:
    print("MPS 不可用（這台不是 Mac 或沒有啟用 MPS）")
