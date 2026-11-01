from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 保持输入形状
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight

        if len(original_shape) > 2:
            x = x.reshape(original_shape)
        return x


class StableSinkhornKnopp(nn.Module):
    """稳定的Sinkhorn-Knopp算法."""

    def __init__(self, num_iterations=20, epsilon=1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.epsilon = epsilon

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: 形状为 (..., n, n) 的矩阵

        Returns:
            双随机矩阵，形状与M相同.
        """
        # 确保正数
        X = torch.exp(M)

        # 迭代行和列归一化
        for _ in range(self.num_iterations):
            # 行归一化
            row_sum = X.sum(dim=-1, keepdim=True) + self.epsilon
            X = X / row_sum

            # 列归一化
            col_sum = X.sum(dim=-2, keepdim=True) + self.epsilon
            X = X / col_sum

        return X


class ManifoldConstrainedHC(nn.Module):
    """流形约束超连接 (mHC) 模块 严格按照论文公式实现.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        use_dynamic_mapping: bool = True,
        use_static_mapping: bool = True,
        sinkhorn_iterations: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.C = hidden_dim // expansion_factor  # 每个流的维度
        self.flat_dim = expansion_factor * self.C  # n×C

        # 动态映射参数 (φ参数)
        if use_dynamic_mapping:
            self.phi_pre = nn.Linear(self.flat_dim, expansion_factor, bias=False)
            self.phi_post = nn.Linear(self.flat_dim, expansion_factor, bias=False)
            self.phi_res = nn.Linear(self.flat_dim, expansion_factor * expansion_factor, bias=False)

        # 静态映射偏置 (b参数)
        if use_static_mapping:
            self.b_pre = nn.Parameter(torch.zeros(1, expansion_factor))
            self.b_post = nn.Parameter(torch.zeros(1, expansion_factor))
            self.b_res = nn.Parameter(torch.zeros(1, expansion_factor * expansion_factor))

        # 可学习的门控因子 (α参数)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        # RMSNorm
        self.rms_norm = RMSNorm(self.flat_dim)

        # Sinkhorn-Knopp投影
        self.sinkhorn = StableSinkhornKnopp(num_iterations=sinkhorn_iterations)

        self.use_dynamic = use_dynamic_mapping
        self.use_static = use_static_mapping

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化参数."""
        if self.use_dynamic:
            nn.init.normal_(self.phi_pre.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.phi_post.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.phi_res.weight, mean=0.0, std=0.01)

    def compute_mappings(self, x: torch.Tensor):
        """计算H_pre, H_post, H_res映射.

        Args:
            x: 输入张量，形状为 (..., n, C)

        Returns:
            tuple: (H_pre, H_post, H_res)
        """
        # 保存原始形状
        original_shape = x.shape
        batch_dims = original_shape[:-2]

        # 展平批次维度
        x_flat = x.reshape(-1, self.expansion_factor, self.C)
        batch_size_flat = x_flat.shape[0]

        # 展平输入：保留完整上下文信息
        x_flat_vec = x_flat.reshape(batch_size_flat, -1)  # [batch_size_flat, n*C]
        x_norm = self.rms_norm(x_flat_vec)

        # 计算原始映射
        if self.use_dynamic:
            dynamic_pre = self.phi_pre(x_norm)  # [batch_size_flat, n]
            dynamic_post = self.phi_post(x_norm)  # [batch_size_flat, n]
            dynamic_res = self.phi_res(x_norm)  # [batch_size_flat, n*n]
            dynamic_res = dynamic_res.reshape(batch_size_flat, self.expansion_factor, self.expansion_factor)
        else:
            dynamic_pre = torch.zeros(batch_size_flat, self.expansion_factor, device=x.device)
            dynamic_post = torch.zeros(batch_size_flat, self.expansion_factor, device=x.device)
            dynamic_res = torch.zeros(batch_size_flat, self.expansion_factor, self.expansion_factor, device=x.device)

        if self.use_static:
            static_pre = self.b_pre.expand(batch_size_flat, -1)
            static_post = self.b_post.expand(batch_size_flat, -1)
            static_res = self.b_res.expand(batch_size_flat, -1)
            static_res = static_res.reshape(batch_size_flat, self.expansion_factor, self.expansion_factor)
        else:
            static_pre = torch.zeros(batch_size_flat, self.expansion_factor, device=x.device)
            static_post = torch.zeros(batch_size_flat, self.expansion_factor, device=x.device)
            static_res = torch.zeros(batch_size_flat, self.expansion_factor, self.expansion_factor, device=x.device)

        # 组合动态和静态部分
        H_pre_raw = self.alpha_pre * dynamic_pre + static_pre
        H_post_raw = self.alpha_post * dynamic_post + static_post
        H_res_raw = self.alpha_res * dynamic_res + static_res

        # 应用约束
        H_pre = torch.sigmoid(H_pre_raw)  # 非负约束
        H_post = 2.0 * torch.sigmoid(H_post_raw)  # 缩放的非负约束
        H_res = self.sinkhorn(H_res_raw)  # 双随机约束

        # 恢复原始批次维度
        H_pre = H_pre.reshape(*batch_dims, self.expansion_factor)
        H_post = H_post.reshape(*batch_dims, self.expansion_factor)
        H_res = H_res.reshape(*batch_dims, self.expansion_factor, self.expansion_factor)

        return H_pre, H_post, H_res

    def forward(self, x: torch.Tensor, layer_fn: callable) -> torch.Tensor:
        """mHC前向传播.

        Args:
            x: 输入张量，形状 (..., n, C)
            layer_fn: 层函数，接受 (..., C) 输入，返回 (..., C) 输出

        Returns:
            输出张量，形状 (..., n, C)
        """
        # 计算映射
        H_pre, H_post, H_res = self.compute_mappings(x)

        # 计算层函数F的输入：H_pre * x
        # 使用einsum进行批量矩阵乘法
        x_weighted = torch.einsum("...n,...nc->...c", H_pre, x)

        # 应用层函数F
        f_output = layer_fn(x_weighted)  # [..., C]

        # 计算残差部分：H_res * x
        residual = torch.einsum("...nm,...mc->...nc", H_res, x)  # [..., n, C]

        # 计算post部分：H_post^T * f_output
        post = torch.einsum("...n,...c->...nc", H_post, f_output)  # [..., n, C]

        # 最终输出
        output = residual + post

        return output


class MultiHeadAttention(nn.Module):
    """标准的多头注意力机制."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 线性变换
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        batch_size, seq_len, hidden_dim = x.shape

        # 线性变换并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class FeedForwardNetwork(nn.Module):
    """前馈神经网络."""

    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, ffn_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # SwiGLU激活函数
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))


class MHCTransformerBlock(nn.Module):
    """集成mHC的Transformer块 严格按照论文架构设计.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        use_mhc_for_attn: bool = True,
        use_mhc_for_ffn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.C = hidden_dim // expansion_factor  # 每个流的维度

        # 是否在注意力和FFN中使用mHC
        self.use_mhc_for_attn = use_mhc_for_attn
        self.use_mhc_for_ffn = use_mhc_for_ffn

        if use_mhc_for_attn:
            # mHC用于注意力层
            self.mhc_attn = ManifoldConstrainedHC(hidden_dim=hidden_dim, expansion_factor=expansion_factor)

            # 定义注意力层函数（处理C维度）
            self.attn_layer = MultiHeadAttention(
                hidden_dim=self.C,  # 注意：这里是C维度
                num_heads=max(1, num_heads // expansion_factor),
                dropout=dropout,
            )
        else:
            # 标准残差连接 + 注意力
            self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
            self.attn_norm = nn.LayerNorm(hidden_dim)

        if use_mhc_for_ffn:
            # mHC用于FFN层
            self.mhc_ffn = ManifoldConstrainedHC(hidden_dim=hidden_dim, expansion_factor=expansion_factor)

            # 定义FFN层函数（处理C维度）
            self.ffn_layer = FeedForwardNetwork(
                hidden_dim=self.C, ffn_dim=ffn_dim // expansion_factor if ffn_dim else None, dropout=dropout
            )
        else:
            # 标准残差连接 + FFN
            self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, dropout)
            self.ffn_norm = nn.LayerNorm(hidden_dim)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Transformer块前向传播.

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力掩码，形状 (batch_size, seq_len, seq_len)

        Returns:
            输出张量，形状 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 重塑为多流格式: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, n, C)
        n, C = self.expansion_factor, self.C
        x_multi = x.reshape(batch_size, seq_len, n, C)

        # 第一部分：注意力层
        if self.use_mhc_for_attn:
            # 使用mHC的注意力

            # 重塑以处理序列维度
            x_flat = x_multi.reshape(batch_size * seq_len, n, C)

            # 定义注意力层函数（处理展平的序列）
            def attn_layer_fn(x_single: torch.Tensor) -> torch.Tensor:
                # x_single: (batch_size*seq_len, C)
                x_reshaped = x_single.reshape(batch_size, seq_len, C)

                # 如果有注意力掩码，需要调整
                if attention_mask is not None:
                    # 注意力层使用的是C维度的掩码
                    attn_output = self.attn_layer(x_reshaped)
                else:
                    attn_output = self.attn_layer(x_reshaped)

                return attn_output.reshape(batch_size * seq_len, C)

            # 应用mHC
            x_multi_flat = self.mhc_attn(x_flat, attn_layer_fn)
            x_multi = x_multi_flat.reshape(batch_size, seq_len, n, C)

        else:
            # 标准残差连接 + 注意力
            attn_output = self.attn(x, attention_mask)
            x = x + self.dropout(attn_output)
            x = self.attn_norm(x)
            x_multi = x.reshape(batch_size, seq_len, n, C)

        # 合并流并应用归一化
        x = x_multi.reshape(batch_size, seq_len, hidden_dim)
        x = self.norm1(x)

        # 第二部分：FFN层
        x_multi = x.reshape(batch_size, seq_len, n, C)

        if self.use_mhc_for_ffn:
            # 使用mHC的FFN

            # 重塑以处理序列维度
            x_flat = x_multi.reshape(batch_size * seq_len, n, C)

            # 定义FFN层函数
            def ffn_layer_fn(x_single: torch.Tensor) -> torch.Tensor:
                # x_single: (batch_size*seq_len, C)
                return self.ffn_layer(x_single)

            # 应用mHC
            x_multi_flat = self.mhc_ffn(x_flat, ffn_layer_fn)
            x_multi = x_multi_flat.reshape(batch_size, seq_len, n, C)

        else:
            # 标准残差连接 + FFN
            x = x_multi.reshape(batch_size, seq_len, hidden_dim)
            ffn_output = self.ffn(x)
            x = x + self.dropout(ffn_output)
            x = self.ffn_norm(x)
            x_multi = x.reshape(batch_size, seq_len, n, C)

        # 合并流并应用最终归一化
        x = x_multi.reshape(batch_size, seq_len, hidden_dim)
        x = self.norm2(x)

        return x


# 测试函数
def test_mhc_transformer():
    """测试mHC Transformer模块."""
    batch_size = 2
    seq_len = 16000
    hidden_dim = 512
    num_heads = 8
    expansion_factor = 4

    print("=" * 60)
    print("测试mHC Transformer模块")
    print("=" * 60)

    # 1. 测试mHC模块
    print("\n1. 测试mHC模块:")
    mhc = ManifoldConstrainedHC(hidden_dim=hidden_dim, expansion_factor=expansion_factor)

    # 创建模拟的层函数
    layer_fn = nn.Sequential(
        nn.Linear(hidden_dim // expansion_factor, hidden_dim // expansion_factor * 4),
        nn.GELU(),
        nn.Linear(hidden_dim // expansion_factor * 4, hidden_dim // expansion_factor),
    )

    # 创建输入
    x = torch.randn(batch_size * seq_len, expansion_factor, hidden_dim // expansion_factor)

    # 前向传播
    output = mhc(x, layer_fn)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  mHC参数数量: {sum(p.numel() for p in mhc.parameters()):,}")

    # 测试双随机约束
    _H_pre, _H_post, H_res = mhc.compute_mappings(x[:1])  # 只测试一个样本

    print("\n  双随机性检查:")
    print(f"  行和范围: [{H_res.sum(dim=-1).min():.4f}, {H_res.sum(dim=-1).max():.4f}]")
    print(f"  列和范围: [{H_res.sum(dim=-2).min():.4f}, {H_res.sum(dim=-2).max():.4f}]")

    # 2. 测试mHC Transformer块
    print("\n2. 测试mHC Transformer块:")
    transformer_block = MHCTransformerBlock(
        hidden_dim=hidden_dim, num_heads=num_heads, expansion_factor=expansion_factor
    )

    x_transformer = torch.randn(batch_size, seq_len, hidden_dim)
    output_transformer = transformer_block(x_transformer)

    print(f"  输入形状: {x_transformer.shape}")
    print(f"  输出形状: {output_transformer.shape}")
    print(f"  块参数数量: {sum(p.numel() for p in transformer_block.parameters()):,}")


if __name__ == "__main__":
    # 运行测试
    test_mhc_transformer()
    print("\n\n")
