import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
    """

    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        # predict 是 logits（任意实数），需要先 sigmoid
        predict = torch.sigmoid(predict)

        # 展平 (N, H, W) or (N, C, H, W)
        predict = predict.contiguous().view(num, -1)
        target = target.contiguous().view(num, -1)

        # Soft Dice
        intersection = (predict * target).sum(dim=1)  # sum (p * g)
        union = predict.sum(dim=1) + target.sum(dim=1)  # |p| + |g|

        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)

        # loss = 1 - dice
        loss = 1 - dice.mean()

        return loss


if __name__ == "__main__":

    fake_out = torch.tensor([[7, 7, -5, -5]], dtype=torch.float32)
    fake_label = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    loss_f = DiceLoss()
    loss = loss_f(fake_out, fake_label)

    print(loss)




