---
title: 首页
description: Qwen3-30B-A3B 在 4×RTX A6000 上的批次不变性实测
---

<div class="hero" markdown>

<p class="hero-kicker">vLLM · BATCH INVARIANCE · 实测报告</p>

# 批次不变性，<br>在 4×A6000 上到底成不成立？

<p class="hero-sub">Qwen3-30B-A3B · vLLM 0.23.0 · TP=4 · 32 条 prompt 逐条位级比较</p>

[一页结论](00_index.md){ .md-button .md-button--primary }
[看边界实验](06_boundary_results.md){ .md-button }
[部署建议](08_deployment.md){ .md-button }

</div>

<div class="stat-grid" markdown>

<div class="stat" markdown>
<span class="stat-num">32/32</span>
<span class="stat-label">compiled 模式<br>32 条 prompt 全部位级一致</span>
</div>

<div class="stat" markdown>
<span class="stat-num">1/32</span>
<span class="stat-label">eager 模式<br>只有 5-token target 通过</span>
</div>

<div class="stat" markdown>
<span class="stat-num">255 / 256</span>
<span class="stat-label">eager decode 的<br>批大小边界</span>
</div>

<div class="stat" markdown>
<span class="stat-num">2–3×</span>
<span class="stat-label">compiled 相对 eager<br>的实测吞吐优势</span>
</div>

</div>

## 先看结论

**在这套硬件上，compiled 模式（`torch.compile` + CUDA Graphs，不加
`--enforce-eager`）是唯一通过全部进程内测试的批次不变性配置。**
eager 模式并非“开了 `VLLM_BATCH_INVARIANT=1` 就成立”，它的 BI 只在很小、
很窄的形状内有效；32 条普通 prompt 中只有那条 5-token target 一致。

另一个背景是：A6000 是 Ampere（SM86），而 vLLM 官方的 tuned matmul 表只覆盖
Ada / Hopper / Blackwell。它名义上满足 CC≥8.0，却不在官方 tuned 路径里。
这与“换 attention 后端、换 MoE 后端、甚至换稠密模型都不能消除 eager 的
形状相关失败”相符。

!!! warning "v2 更正声明"

    v1 报告里“短 prompt 位级一致”的结论，只对测试中那一条 5-token target
    成立。扩展到 32 条普通 prompt 后，eager 有 **31/32** 不一致，compiled 是
    **32/32**。本网站以 v2 为准，v1 只作为历史记录保留。

## 从哪一篇开始

<div class="card-grid" markdown>

<div class="card" markdown>
### :material-format-list-checks: 要做部署决策

先读一页结论，再对照部署清单选模式。

[打开一页结论 →](00_index.md){ .card-link }
</div>

<div class="card" markdown>
### :material-table-large: 要看实验数据

表 I–Q：批大小扫描、32-prompt、并发、256 边界、decode vs re-feed。

[打开边界实验 →](06_boundary_results.md){ .card-link }
</div>

<div class="card" markdown>
### :material-lightbulb-on-outline: 要理解原因

F1–F12：每条发现都给出事实、证据与解释强度。

[打开关键发现 →](07_findings.md){ .card-link }
</div>

<div class="card" markdown>
### :material-rocket-launch-outline: 要直接上线

推荐配置、固定编译产物、发布前逐 prompt 验证清单。

[打开部署建议 →](08_deployment.md){ .card-link }
</div>

</div>

## 这次测了什么

- **模型**：`Qwen/Qwen3-30B-A3B`，BF16，MoE，TP=4，`max_model_len=8192`；
- **硬件**：4 × NVIDIA RTX A6000 48 GB（SM86），PCIe，无 InfiniBand；
- **软件**：vLLM 0.23.0+cu129，PyTorch 2.11.0+cu129，CUDA 12.9，NCCL 2.28.9；
- **口径**：用 IEEE-754 位模式比较 logprob，而不是比 token id 或文本；
- **范围**：离线 `LLM.generate` 与在线 HTTP 服务都测；同一进程内、跨 warm-cache
  进程、32 路并发、突发、长 prompt、prefix cache 都覆盖。

想看完整实验矩阵和原始 JSON，请从[复现与文件清单](10_reproduction.md)进入。

## 结论

**要 BI：compiled，固定产物，逐 prompt 验证；eager 只在你能证明的小形状内使用。**
