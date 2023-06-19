# 第一种: 对特定网络模块的剪枝(Pruning Model).

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1: 图像的输入通道(1是黑白图像), 6: 输出通道, 3x3: 卷积核的尺寸
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是经历卷积操作后的图片尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device=device)

module = model.conv1

print(list(module.named_parameters()))
weights_total = sum(p.numel() for p in module.parameters())
# print("Total number of parameters: ", weights_total)

# 第一个参数: module, 代表要进行剪枝的特定模块, 之前我们已经制定了module=model.conv1,
#             说明这里要对第一个卷积层执行剪枝.
# 第二个参数: name, 指定要对选中的模块中的哪些参数执行剪枝.
#             这里设定为name="weight", 意味着对连接网络中的weight剪枝, 而不对bias剪枝.
# 第三个参数: amount, 指定要对模型中多大比例的参数执行剪枝.
#             amount是一个介于0.0-1.0的float数值, 或者一个正整数指定剪裁掉多少条连接边.
print(model.state_dict().keys())
prune.random_unstructured(module, name="weight", amount=0.3)
weights_total = sum(p.numel() for p in module.parameters())
# print("Total number of parameters: ", weights_total)
print(list(module.named_parameters()))
print(list(module.named_buffers()))

print("--------------------")
print(module.weight)
print(model.state_dict().keys())
#
# # 对模型进行剪枝操作, 分别在weight和bias上剪枝
# module = model.conv1
# prune.random_unstructured(module, name="weight", amount=0.3)
# prune.l1_unstructured(module, name="bias", amount=3)
#
# # 再将剪枝后的模型的状态字典打印出来
# print(model.state_dict().keys())
#
# # 对模型执行剪枝remove操作.
# # 通过module中的参数weight_orig和weight_mask进行剪枝, 本质上属于置零遮掩, 让权重连接失效.
# # 具体怎么计算取决于_forward_pre_hooks函数.
# # 这个remove是无法undo的, 也就是说一旦执行就是对模型参数的永久改变.
#
# # 打印剪枝后的模型参数
# print(list(module.named_parameters()))
# print('*'*50)
#
# # 打印剪枝后的模型mask buffers参数
# print(list(module.named_buffers()))
# print('*'*50)
#
# # 打印剪枝后的模型weight属性值
# print(module.weight)
# print('*'*50)
#
# # 打印模型的_forward_pre_hooks
# print(module._forward_pre_hooks)
# print('*'*50)
#
# # 执行剪枝永久化操作remove
# prune.remove(module, 'weight')
# print('*'*50)
#
# # remove后再次打印模型参数
# print(list(module.named_parameters()))
# print('*'*50)
#
# # remove后再次打印模型mask buffers参数
# print(list(module.named_buffers()))
# print('*'*50)
#
# # remove后再次打印模型的_forward_pre_hooks
# print(module._forward_pre_hooks)

# 对模型的weight执行remove操作后, 模型参数集合中只剩下bias_orig了,
# weight_orig消失, 变成了weight, 说明针对weight的剪枝已经永久化生效.
# 对于named_buffers张量打印可以看出, 只剩下bias_mask了,
# 因为针对weight做掩码的weight_mask已经生效完毕, 不再需要保留了.
# 同理, 在_forward_pre_hooks中也只剩下针对bias做剪枝的函数了.