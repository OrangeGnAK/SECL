
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

from losses import FastSupCon

class SupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [N, C], labels: [N]
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = 1.0 - torch.eye(features.size(0), device=features.device)
        mask = mask * logits_mask
        
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return -mean_log_prob_pos.mean()


def profile_loss(loss_type, loss_fn, features, labels, num_iters=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if loss_type == "linear":
        feat_in = features.unsqueeze(0)
        lab_in = labels.unsqueeze(0)
    else:
        feat_in = features
        lab_in = labels

    # Warmup
    for _ in range(2):
        loss = loss_fn(feat_in, lab_in)
        loss.backward()
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iters):
        loss = loss_fn(feat_in, lab_in)
        loss.backward()
        
    torch.cuda.synchronize()
    avg_time_ms = ((time.time() - start_time) / num_iters) * 1000
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return avg_time_ms, peak_memory_mb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dim = 128
    num_classes = 10

    pixel_range = range(100, 500000, 1000)
    
    n_axis = []
    supcon_times, supcon_mems = [], []
    linear_times, linear_mems = [], []
    
    supcon_oom_triggered = False
    
    print(f"{'N pixels':<10} | {'SupCon Time':<12} | {'SupCon Mem':<11} | {'Linear Time':<12} | {'Linear Mem':<11}")
    print("-" * 65)

    for N in pixel_range:
        n_axis.append(N)
        
        features = torch.randn(N, feat_dim, device=device, requires_grad=True)
        labels = torch.randint(0, num_classes, (N,), device=device)
        
        if not supcon_oom_triggered:
            try:
                t, m = profile_loss("standard", SupCon().to(device), features, labels)
                supcon_times.append(t)
                supcon_mems.append(m)
                supcon_t_str, supcon_m_str = f"{t:.1f}ms", f"{m:.1f}MB"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    supcon_oom_triggered = True
                    print(f"\n[!!!] SupCon caught OOM at N = {N} pixels\n")
                    supcon_times.append(None)
                    supcon_mems.append(None)
                    supcon_t_str, supcon_m_str = "OOM", "OOM"
                else:
                    raise e
        else:
            supcon_times.append(None)
            supcon_mems.append(None)
            supcon_t_str, supcon_m_str = "OOM", "OOM"
            
        t_lin, m_lin = profile_loss("linear", FastSupCon(num_classes=num_classes).to(device), features, labels)
        linear_times.append(t_lin)
        linear_mems.append(m_lin)
        
        print(f"{N:<10} | {supcon_t_str:<12} | {supcon_m_str:<11} | {t_lin:.1f}ms     | {m_lin:.1f}MB")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    valid_supcon_time = [x for x in supcon_times if x is not None]
    ax1.plot(n_axis[:len(valid_supcon_time)], valid_supcon_time, label='SupCon O(N^2)', color='red', linewidth=2)
    ax1.plot(n_axis, linear_times, label='SECL O(N)', color='green', linewidth=2)
    ax1.set_title('Time (ms)')
    ax1.set_xlabel('Pixels (N)')
    ax1.set_ylabel('Execution time (ms)')
    ax1.legend()
    ax1.grid(True)
    
    # Memory plot 
    valid_supcon_mem = [x for x in supcon_mems if x is not None]
    ax2.plot(n_axis[:len(valid_supcon_mem)], valid_supcon_mem, label='SupCon O(N^2)', color='red', linewidth=2)
    ax2.plot(n_axis, linear_mems, label='SECL O(N)', color='green', linewidth=2)
    
    if supcon_oom_triggered:
        oom_n = n_axis[len(valid_supcon_mem)]
        ax2.axvline(x=oom_n, color='darkred', linestyle='--', label=f'SupCon OOM stage ({oom_n} px)')
        ax1.axvline(x=oom_n, color='darkred', linestyle='--')
        
    ax2.set_title('Peak memory consumption (MB)')
    ax2.set_xlabel('Number of pixels (N)')
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_benchmarks.png', dpi=300)
    plt.show()
    print("Plot saved 'loss_benchmarks.png'")
