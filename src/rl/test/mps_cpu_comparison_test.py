import torch, time

# === 模擬你的專案維度 ===
N, F, K = 300, 11, 15
L = N * K   # sequence length
d = 64      # embedding dim

print(f"Benchmarking Attention: L={L}, d={d}, repeat=500")

# 建立隨機 tensor
q = torch.randn(1, L, d)
k = torch.randn(1, L, d)
v = torch.randn(1, L, d)

def run_attention(device, repeat=500):
    q_d, k_d, v_d = q.to(device), k.to(device), v.to(device)
    if device.type == "mps":
        torch.mps.synchronize()

    # 預熱一次（避免第一次初始化干擾）
    _ = (q_d @ k_d.transpose(-2, -1)) @ v_d
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.time()
    for _ in range(repeat):
        attn = (q_d @ k_d.transpose(-2, -1)) @ v_d
    if device.type == "mps":
        torch.mps.synchronize()
    total = time.time() - start
    return total, total / repeat

# === CPU 測試 ===
cpu_total, cpu_avg = run_attention(torch.device("cpu"))
print(f"CPU: {cpu_total:.4f} 秒 (總共500次)，平均 {cpu_avg:.6f} 秒/次")

# === MPS 測試（如果可用） ===
if torch.backends.mps.is_available():
    mps_total, mps_avg = run_attention(torch.device("mps"))
    print(f"MPS: {mps_total:.4f} 秒 (總共500次)，平均 {mps_avg:.6f} 秒/次")
else:
    print("MPS 不可用（這台不是 Mac 或沒有啟用 MPS）")
