import torch
import torch.nn as nn


def calculate_mse_loss(vec1, vec2):
    # 将向量展平以匹配 MSELoss 的要求
    vec1 = vec1.view(vec1.size(0), -1)
    vec2 = vec2.view(vec2.size(0), -1)

    # 创建 MSELoss 计算对象
    criterion = nn.MSELoss(reduction='none')

    # 计算每个批次的均方误差损失
    loss = criterion(vec1, vec2).mean(dim=1)

    return loss


# 示例用法
vec1 = torch.randn(8, 512, 1, 53)
vec2 = torch.randn(8, 512, 1, 53)

mse_loss = calculate_mse_loss(vec1, vec2)
print(mse_loss)