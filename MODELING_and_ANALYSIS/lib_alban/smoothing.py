### Original code taken from https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/ and modified for our purpose.
import torch
import torch.nn as nn
import torch.nn.functional as F



def find_local_patch(x, patch_size):
    N, C, H, W = x.shape
    x_unfold = F.unfold(
        x, kernel_size=(patch_size, patch_size), padding=(patch_size // 2, patch_size // 2), stride=(1, 1)
    )

    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        super(WeightedAverage, self).__init__()

    def forward(self, x_l, patch_size=3, alpha=1, scale_factor=1):
        # alpha=0: less smooth; alpha=inf: smoother
        x_l = F.interpolate(x_l, scale_factor=scale_factor)
        local_l = find_local_patch(x_l, patch_size)
        local_difference_l = (local_l - x_l ** 2)
        correlation = nn.functional.softmax(-1 * local_difference_l / alpha, dim=1)

        return torch.cat(
            (
                torch.sum(correlation * local_l, dim=1, keepdim=True),
            ),
            1,
        )

def batch_RGB2PCA(x):
    #M = np.array([[ 0.56677561,  0.71836896,  -0.40189701],[ 0.58101187,  0.02624069,  0.81321349],[ 0.58406993,  -0.69441022,  -0.41888652]])
    M = torch.tensor([[ 0.66666,  1,  -0.5],[ 0.66666,  0,  1],[ 0.66666,  -1,  -0.5]]).cuda()
    x = torch.transpose(x, 1,0)
    x = torch.reshape(x, (3,-1))
    x = torch.matmul(M.T, x - 0.5)
    x = torch.reshape(x, (3,-1,256,256))
    return torch.transpose(x, 1,0)

def batch_PCA2RGB(x):
    #M = np.array([[ 0.56677561,  0.71836896,  -0.40189701],[ 0.58101187,  0.02624069,  0.81321349],[ 0.58406993,  -0.69441022,  -0.41888652]])
    M = torch.inverse(torch.tensor([[ 0.66666,  1,  -0.5],[ 0.66666,  0,  1],[ 0.66666,  -1,  -0.5]])).cuda()
    x = torch.transpose(x, 1,0)
    x = torch.reshape(x, (3,-1))
    x = torch.matmul(M.T, x) + 0.5
    x = torch.reshape(x, (3,-1,256,256))
    return torch.transpose(x, 1,0)

class WeightedAverage_color(nn.Module):
    """
    smooth the image according to the color distance in the LAB space
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color, self).__init__()

    def forward(self, x_rgb, x_rgb_predict, patch_size=3, alpha=1, scale_factor=1):
        """ alpha=0: less smooth; alpha=inf: smoother """
        #x_rgb = F.interpolate(x_rgb, scale_factor=scale_factor)

        x_pca = batch_RGB2PCA(x_rgb) # conversion to opponent space to consider chromatic components only
        x_pca_predict = batch_RGB2PCA(x_rgb_predict)
        #l = x_pca[:, 0:1, :, :]
        a = x_pca[:, 1:2, :, :]
        b = x_pca[:, 2:3, :, :]
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_l_predict = find_local_patch(x_pca_predict[:, 0:1, :, :], patch_size)
        local_a_predict = find_local_patch(x_pca_predict[:, 1:2, :, :], patch_size)
        local_b_predict = find_local_patch(x_pca_predict[:, 2:3, :, :], patch_size)

        local_color_difference = (local_a - a) ** 2 + (local_b - b) ** 2
        correlation = nn.functional.softmax(
            -1 * local_color_difference / alpha, dim=1
        )  # so that sum of weights equal to 1
        smoothed_img = torch.cat(
            (
                x_pca_predict[:, 0:1, :, :],
                #torch.sum(correlation * local_l_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_a_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_b_predict, dim=1, keepdim=True),
            ),
            1,
        )
        #import pdb; pdb.set_trace()
        return batch_PCA2RGB(smoothed_img)