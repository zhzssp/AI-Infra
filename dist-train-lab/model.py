"""主角模型：`tiny_mlp` 的同构放大版。

全仓库的图级主角是 `tiny_mlp`：

    x ──Gemm(W[out,in], b)──▶ Relu ──▶ Add(bias2) ──▶ y

它在 `onnx-delegate-lab` 是 ONNX 图、在 `mlir-toy-dialect/examples/upstream` 是
linalg IR、在 `iree-lab` 被真编真跑、在 `tvm-fatbin-lab` 是 Relay 图。但它只有
64 次浮点运算——单张 RTX 5090 跑完它约需 6e-13 秒，**比硬件小十二个数量级**。

本 lab 把它放大到 GPU 说得上话的尺度。放大后的单元是 Transformer 的 FFN block：

    x ──LayerNorm──▶ fc1 ──GELU──▶ fc2 ──▶ Add(x) ──▶ y
                      │      │              │
                      │      │              └─ 对应 tiny_mlp 的 Add(bias2)
                      │      └──────────────── 对应 tiny_mlp 的 Relu
                      └─────────────────────── 对应 tiny_mlp 的 Gemm(W, b)

**诚实地说，这不是严格同构**：FFN 比 tiny_mlp 多了一个降维矩阵 `fc2` 和一个
`LayerNorm`。但骨架「Gemm → 逐元素激活 → 逐元素加」完全一致，而且放大的正是
那根骨架——所以你在前五个 lab 里对这张图建立的所有直觉（哪些算子能融合、
归约维在哪、权重按 [out,in] 存）在这里继续成立。

`TinyMlp` 保留了**原尺寸的主角**，用作数值锚点：喂 x=[1,2,3] 必须得到
[2.5, 3.5, 4.5, 7.5]，与另外四个 lab 逐位一致。跑 `python model.py` 可验证。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 原尺寸主角：跨 lab 的数值锚点
# --------------------------------------------------------------------------

TINY_W = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
TINY_B = 0.5
TINY_BIAS2 = 1.0
TINY_EXPECTED = [2.5, 3.5, 4.5, 7.5]  # 喂 x=[1,2,3] 时四个系统都该给出这个


class TinyMlp(nn.Module):
    """与 onnx/iree/tvm/mlir 四个 lab 里那张图**完全相同**的模型。

    nn.Linear 的 weight 就是 [out, in] 布局——这正是 ONNX Gemm 要写 transB=1、
    linalg 要写 affine_map<(m,n,k)->(n,k)> 的原因。三处说的是同一件事。
    """

    def __init__(self) -> None:
        super().__init__()
        self.gemm = nn.Linear(3, 4)
        with torch.no_grad():
            self.gemm.weight.copy_(torch.tensor(TINY_W))
            self.gemm.bias.fill_(TINY_B)
        self.register_buffer("bias2", torch.full((4,), TINY_BIAS2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.gemm(x)) + self.bias2


def check_tiny_mlp(atol: float = 1e-6) -> tuple[bool, list[float]]:
    """跨 lab 数值一致性自检。"""
    with torch.no_grad():
        y = TinyMlp()(torch.tensor([[1.0, 2.0, 3.0]]))
    got = y.reshape(-1).tolist()
    ok = all(abs(a - b) < atol for a, b in zip(got, TINY_EXPECTED))
    return ok, got


# --------------------------------------------------------------------------
# 放大版：参数量随 hidden 平方增长
# --------------------------------------------------------------------------


class FFNBlock(nn.Module):
    """Transformer 的 FFN 结构，占 Transformer 参数量的大头。

    刻意去掉 attention：它的形状与 mask 会把显存账搅浑，而本 lab 要考的是
    「16Φ 的构成」和「分片切了哪一项」，attention 只会增加噪声。需要真
    attention 时把 nn.MultiheadAttention 加回来即可，结论不变。
    """

    def __init__(self, hidden: int, expand: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, expand * hidden)  # TP 的列切对象
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expand * hidden, hidden)  # TP 的行切对象
        self.use_checkpoint = False

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            # use_reentrant=False 是新版推荐值；reentrant 版对 RNG 与 grad 有额外的坑
            return x + checkpoint(self._body, x, use_reentrant=False)
        return x + self._body(x)


def build_model(hidden: int = 1024, layers: int = 8, expand: int = 4) -> nn.Module:
    blocks = [FFNBlock(hidden, expand) for _ in range(layers)]
    return nn.Sequential(*blocks, nn.LayerNorm(hidden))


def set_activation_checkpointing(model: nn.Module, enabled: bool) -> int:
    """写法 A：直接开 block 内部的 checkpoint。最少侵入，单卡也适用。

    返回被影响的 block 数——为 0 说明没匹配上，这时显存和吞吐都不会变，
    而「显存和吞吐同时不变」正是重计算没生效的典型症状。
    """
    n = 0
    for m in model.modules():
        if isinstance(m, FFNBlock):
            m.use_checkpoint = enabled
            n += 1
    return n


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def params_closed_form(hidden: int, layers: int, expand: int = 4) -> int:
    """闭式解，用来和 count_params 对账，也用来喂 mem_ledger。

    每层 = 2*expand*hidden²（两个 Linear 的权重）
         + (expand+1)*hidden（两个 Linear 的 bias）
         + 2*hidden（LayerNorm 的 weight+bias）
    末尾再加一个 LayerNorm。expand=4 时每层约 8*hidden²。
    """
    per_layer = 2 * expand * hidden * hidden + (expand + 1) * hidden + 2 * hidden
    return layers * per_layer + 2 * hidden


def synthetic_batch(
    batch: int, seq: int, hidden: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """合成数据。本 lab 不关心收敛，但**各 rank 必须拿到不同数据**——
    否则 all_reduce 前后梯度相同，等于什么都没验证。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, seq, hidden, generator=g).to(device)
    y = torch.randn(batch, seq, hidden, generator=g).to(device)
    return x, y


if __name__ == "__main__":
    ok, got = check_tiny_mlp()
    print(f"tiny_mlp 数值锚点: {got}")
    print(f"  期望 {TINY_EXPECTED} → {'一致 ✔' if ok else '不一致 ✘'}")
    print()
    for hidden, layers in [(256, 4), (1024, 8), (2048, 12), (4096, 12)]:
        m = build_model(hidden, layers)
        actual, closed = count_params(m), params_closed_form(hidden, layers)
        err = abs(actual - closed) / closed
        print(
            f"hidden={hidden:5d} layers={layers:3d}  "
            f"Φ={actual/1e6:9.2f}M  16Φ={actual*16/2**30:7.2f} GiB  "
            f"闭式解误差={err:.2e}"
        )
