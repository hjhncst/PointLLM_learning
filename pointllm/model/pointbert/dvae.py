import torch.nn as nn
import torch
import torch.nn.functional as F
from . import misc

# from knn_cuda import KNN

# knn = KNN(k=4, transpose_mode=False)


class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        '''
        K has to be 16
        '''
        # 输入转换层，将输入特征从encoder_channel转换为128维
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        # 定义了四个二维卷积层，每个层都包含一个卷积操作、一个组归一化操作和一个 LeakyReLU 激活函数
        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 1024),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        
        # 输出层，将特征从1024维转换为output_channel维
        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, output_channel),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np


        # coor表示坐标，x表示特征，bs表示批量大小，np表示点数，c表示特征维度
        k = 4  # 设置k近邻的k值为4
        batch_size = x_k.size(0)  # 获取批量大小
        num_points_k = x_k.size(2)  # 获取k点的点数
        num_points_q = x_q.size(2)  # 获取q点的点数

        with torch.no_grad():  # 不计算梯度，用于推理阶段
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3

        # 将坐标和特征从 B G C 转换为 B C N 和 B 3 N
        # bs 3 N   bs C N
        feature_list = []
        coor = coor.transpose(1, 2).contiguous()  # B 3 N
        f = f.transpose(1, 2).contiguous()  # B C N
        f = self.input_trans(f)  # B 128 N

        # 获取图特征并进行第一次卷积
        f = self.get_graph_feature(coor, f, coor, f)  # B 256 N k
        f = self.layer1(f)  # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 256 N
        feature_list.append(f)

        # 获取图特征并进行第二次卷积
        f = self.get_graph_feature(coor, f, coor, f)  # B 512 N k
        f = self.layer2(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        # 获取图特征并进行第三次卷积
        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer3(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        # 获取图特征并进行第四次卷积
        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer4(f)  # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 1024 N
        feature_list.append(f)

        # 将所有特征拼接在一起
        f = torch.cat(feature_list, dim=1)  # B 2304 N

        # 进行最后一次卷积
        f = self.layer5(f)  # B C' N

        # 将特征从 B C' N 转换为 B N C'
        f = f.transpose(-1, -2)

        return f


### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    输入参数:
        nsample: 每个局部区域内的最大采样点数
        xyz: 所有点的集合，形状为 [B, N, C]，其中 B 表示批大小，N 表示点的数量，C 表示点的维度
        new_xyz: 查询点，形状为 [B, S, C]，其中 S 表示查询点的数量
    返回:
        group_idx: 分组后的点的索引，形状为 [B, S, nsample]
    """
    # 计算查询点 new_xyz 与所有点 xyz 之间的平方距离
    sqrdists = square_distance(new_xyz, xyz)
    # 在最后一个维度上寻找距离最小的 nsample 个点的索引，largest=False 表示取最小值，sorted=False 表示返回的索引不排序
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    # 返回每个查询点对应的局部区域内的点的索引
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # 获取源点云的批次大小B，点数N，和特征维度C
    B, N, _ = src.shape
    # 获取目标点云的点数M
    _, M, _ = dst.shape
    # 计算源点云和目标点云的内积，得到[B, N, M]的矩阵
    # src: [B, N, C], dst.permute(0, 2, 1): [B, C, M]
    # torch.matmul(src, dst.permute(0, 2, 1)): [B, N, M]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # 计算源点云每个点的平方和，得到[B, N, 1]的矩阵
    # src ** 2: [B, N, C], torch.sum(src ** 2, -1): [B, N]
    # view(B, N, 1): [B, N, 1]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # 计算目标点云每个点的平方和，得到[B, 1, M]的矩阵
    # dst ** 2: [B, M, C], torch.sum(dst ** 2, -1): [B, M]
    # view(B, 1, M): [B, 1, M]
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # 返回每个源点和目标点之间的平方距离，形状为[B, N, M]
    return dist


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3]
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        
        # 第一阶段卷积：将输入的3D点转换为局部特征
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),   # 1维卷积，将点维度从3扩展到128
            nn.BatchNorm1d(128),                         # 进行批量归一化提高训练稳定性
            nn.ReLU(inplace=True),                       # ReLU 激活函数
            nn.Conv1d(128, 256, 1)                        # 再通过1维卷积将通道数提升到256
        )
        
        # 第二阶段卷积：整合局部特征，并生成最终编码特征
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),                       # 输入512通道（拼接后的局部特征和全局特征），输出还是512通道
            nn.BatchNorm1d(512),                         # 批量归一化
            nn.ReLU(inplace=True),                       # ReLU 激活函数
            nn.Conv1d(512, self.encoder_channel, 1)       # 最后一层卷积将通道数转换为encoder_channel
        )

    def forward(self, point_groups):
        '''
            point_groups : 点组数据, 尺寸为 [B, G, N, 3]
            -----------------
            feature_global : 输出全局编码后的特征, 尺寸为 [B, G, encoder_channel]
        '''
        bs, g, n, c = point_groups.shape  # 获取批次大小 B, 点组数量 G, 每组点数 N 和点的维度 C
        point_groups = point_groups.reshape(bs * g, n, c)  # 将点组重构为 [B*G, N, 3]
        
        # 将数据转置后输入 first_conv，转换通道维度，得到局部特征 [B*G, 256, N]
        feature = self.first_conv(point_groups.transpose(2, 1))
        
        # 在每个点组上进行全局最大池化，提取出全局特征 [B*G, 256, 1]
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        
        # 将全局特征扩展到每个点，并与局部特征拼接，构成 [B*G, 512, N]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        
        # 经过第二阶段卷积处理，输出编码后的特征 [B*G, encoder_channel, N]
        feature = self.second_conv(feature)
        
        # 对处理后的特征进行全局最大池化，提取每个点组的全局特征 [B*G, encoder_channel]
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        # 将全局特征 reshape 成 [B, G, encoder_channel]作为最终输出
        return feature_global.reshape(bs, g, self.encoder_channel)


class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2                      # 折叠网格的尺寸，2x2网格
        self.num_coarse = self.num_fine // 4      # 粗略点云数量，必须被4整除
        assert num_fine % 4 == 0

        # 利用三层全连接层（MLP）将全局特征转换成粗略点云
        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),     # 将输入通道转换到1024维
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),                # 保持1024维
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)    # 输出 coarse 点云的坐标，3为每个点的维度
        )
        # 最终卷积，生成细化点云的偏移量
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),   # 融合全局特征、局部折叠种子和初始点特征
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)                           # 输出细化点云的3D坐标偏移量
        )
        # 构造折叠操作中使用的2D采样网格，即 folding_seed
        # 生成一个从 -0.05 到 0.05 的等间距坐标，用于折叠操作
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size)
        a = a.expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1)
        b = b.expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # 形状为 [1, 2, grid_size^2]

    def forward(self, feature_global):
        '''
            输入:
                feature_global : [B, G, C] 全局特征，其中 B 为批量大小, G 为点组数量, C 为特征维度
            输出:
                coarse : [B, G, M, 3] 粗略生成的点云，其中 M=num_coarse
                fine : [B, G, N, 3] 细化生成的点云，其中 N=num_fine
        '''
        bs, g, c = feature_global.shape
        # 将特征展开为一个批次，便于后续处理，形状变为 [B*G, C]
        feature_global = feature_global.reshape(bs * g, c)

        # 利用 MLP 将全局特征转换为粗略点云，形状为 [B*G, num_coarse * 3]
        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)  # [B*G, M, 3]

        # 对 coarse 点云添加一个维度后，在该维度上扩展为折叠网格尺寸，得到每个粗略点对应多个局部初始点
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # [B*G, M, S, 3] 其中 S=grid_size^2
        # 重构到细化点云尺寸，并调整通道维度到前面，结果 [B*G, 3, num_fine]
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1)  # [B*G, 3, N]

        # 扩展 folding_seed 到每个样本和每个粗略点上，结果 shape: [B*G, 2, num_coarse * S]
        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1)  # [B*G, 2, M, S]
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)         # [B*G, 2, N]

        # 扩展全局特征到每个细化点上，结果 shape: [B*G, C, num_fine]
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)  # [B*G, encoder_channel, N]

        # 将全局特征、折叠种子和初始点特征拼接在一起，作为后续卷积运算的输入
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # [B*G, (encoder_channel+2+3), N]

        # 计算粗略点云的中心，将 coarse 展开为细化尺寸对应的中心
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # [B*G, M, S, 3]
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1)         # [B*G, 3, N]

        # 通过 final_conv 得到细化点云的偏移，再加上中心位置得到细化点云坐标
        fine = self.final_conv(feat) + center  # [B*G, 3, N]

        # reshape 输出的细化点云和粗略点云到原始的批次和组数形式
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)           # [B, G, N, 3]
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)                        # [B, G, M, 3]
        return coarse, fine


class DiscreteVAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # 配置各个网络层所需的参数
        self.group_size = config.group_size         # 每组点云的大小
        self.num_group = config.num_group           # 组的数量
        self.encoder_dims = config.encoder_dims     # Encoder 输出的特征通道数
        self.tokens_dims = config.tokens_dims       # 离散tokens的特征维度

        self.decoder_dims = config.decoder_dims     # Decoder 输入的特征通道数
        self.num_tokens = config.num_tokens         # 离散tokens的数量

        # 按组划分点云数据
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # Encoder：对每个局部点组进行编码，输出全局特征
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # 第一个 DGCNN：从encoder输出特征映射到离散tokens的logits
        self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=self.num_tokens)
        # 维护一个 codebook 参数，用于将 tokens 映射到连续表示中（特征嵌入）
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        # 第二个 DGCNN：对离散特征进行进一步处理，得到 decoder 使用的特征
        self.dgcnn_2 = DGCNN(encoder_channel=self.tokens_dims, output_channel=self.decoder_dims)
        # Decoder：利用全局特征生成粗略和细化点云
        self.decoder = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size)
        # 损失函数部分代码被注释，可根据需要启用
        # self.build_loss_func()

    # 以下为损失函数定义
    # def build_loss_func(self):
    #     self.loss_func_cdl1 = ChamferDistanceL1().cuda()
    #     self.loss_func_cdl2 = ChamferDistanceL2().cuda()
    #     self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret, gt):
        # 从网络返回的结果中解包 coarse 和 fine 生成的点云，以及 ground truth 点云（group_gt）
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape  # 获取批次大小和组数

        # 重新组织 coarse、fine 点云形状为 [B*G, 点数, 3]
        coarse = coarse.reshape(bs * g, -1, 3).contiguous()
        fine = fine.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()

        # 计算 coarse 和 fine 点云与 ground truth 之间的 Chamfer 距离（L1版本）
        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)

        # 总体重构损失为 coarse 和 fine 损失之和
        loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):
        # 计算重构损失
        loss_recon = self.recon_loss(ret, gt)
        # 计算 KL 散度
        logits = ret[-1]  # 取出最后的 logits，形状 [B, G, N]
        softmax = F.softmax(logits, dim=-1)
        # 求各组token分布的均值
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        # 创建均匀分布的对数概率
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=gt.device))
        # 基于 token 分布与均匀分布计算 KL 散度
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean',
                            log_target=True)

        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        # 通过 group_divider 将原始点云 inp 划分为局部点组
        neighborhood, center = self.group_divider(inp)
        # Encoder 对每个局部点组进行编码，得到全局特征 [B, G, C]
        logits = self.encoder(neighborhood)
        # 第一个 DGCNN 将 encoder 输出转换为离散tokens的 logits，形状 [B, G, N]
        logits = self.dgcnn_1(logits, center)
        # 使用 Gumbel-Softmax 将 logits 转换成 one-hot 的近似离散分布
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
        # 根据 one-hot 分布，从 codebook 中采样连续表示，进行矩阵乘法实现
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)
        # 第二个 DGCNN 对离散采样后的特征和中心点信息进行进一步处理
        feature = self.dgcnn_2(sampled, center)
        # Decoder 利用处理后的特征生成粗略和细化点云
        coarse, fine = self.decoder(feature)

        # 计算整体点云（粗略和细化）时，将中心点信息加回去，结果包装为 [B, num_points, 3]
        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        # 保证细化点云的点数与 group_size 一致
        assert fine.size(2) == self.group_size
        # 返回的元组包含整体粗略点云、整体细化点云、原始粗略点云、原始细化点云、局部点组、以及 tokens logits
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret