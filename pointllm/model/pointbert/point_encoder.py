import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .dvae import Group
from .dvae import Encoder
from .logger import print_log
from collections import OrderedDict

from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 如果未指定输出通道，默认和输入通道一致
        out_features = out_features or in_features
        # 如果未指定隐藏层通道，默认和输入通道一致
        hidden_features = hidden_features or in_features
        
        # 第一层全连接，将 in_features 转换到 hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活层，默认使用 GELU 激活函数
        self.act = act_layer()
        # 第二层全连接，将隐藏层维度转换到 out_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        # dropout 层，用于防止过拟合
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)   # 通过第一层全连接
        x = self.act(x)   # 激活函数层
        x = self.drop(x)  # dropout 层
        x = self.fc2(x)   # 第二层全连接
        x = self.drop(x)  # 再次进行 dropout
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        实现多头自注意力机制。
        参数:
            dim: 输入特征的维度。
            num_heads: 注意力头的数量，将输入特征均分到各个头中。
            qkv_bias: 是否为 Q, K, V 线性变换添加偏置项。
            qk_scale: 缩放因子，如果未指定则使用 head_dim ** -0.5。
            attn_drop: 注意力权重的 dropout 概率。
            proj_drop: 最终输出投影后的 dropout 概率。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的特征维度
        # 如果没有显式指定 qk_scale，则按每个头的维度计算缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 定义线性层，用于生成查询(Q)、键(K)和值(V)
        # 将输入特征映射到 3 倍的 dim，然后在 forward 中分割为 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力权重的 dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        # 对多头注意力输出进行线性投影
        self.proj = nn.Linear(dim, dim)
        # 最终输出的 dropout 层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播:
            输入 x: [B, N, C]，其中 B 是批量大小，N 是序列长度，C 是特征维度（dim）。
            输出: 经过注意力计算和投影后的特征张量，形状仍为 [B, N, C]。
        """
        B, N, C = x.shape  # 获取批大小、序列长度和特征维度
        # 通过 qkv 层映射输入，并reshape至 [B, N, 3, num_heads, C // num_heads]
        # 然后 permute 维度顺序以便分离出 Q, K, V，各维度为 [B, num_heads, N, C // num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离出查询、键和值

        # 后续的注意力计算代码(未展示)会利用 q, k, v 进行 scaled dot-product attention，
        # 然后对注意力输出进行 dropout 和线性投影操作，最终返回 [B, N, C]
        attn = (
            # ...attention 计算的代码...
        )

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config
        
        self.use_max_pool = use_max_pool # * whethet to max pool the features of different tokens

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.point_dims = config.point_dims
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.point_encoder.'):
                state_dict[k.replace('module.point_encoder.', '')] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print_log("PointBERT's weights are successfully loaded from {}".format(bert_ckpt_path), logger='Transformer')

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f # * B, 1, C(384 + 384)