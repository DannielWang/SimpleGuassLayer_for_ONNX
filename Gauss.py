import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import cv2
import onnxruntime
from PIL import Image


class GaussLayer(nn.Module):
    def __init__(self, channels=3, kernelsize=5, sigma=0):
        super(GaussLayer, self).__init__()
        self.channels = channels
        kx = cv2.getGaussianKernel(kernelsize, sigma)
        ky = cv2.getGaussianKernel(kernelsize, sigma)
        kernel = np.multiply(kx, np.transpose(ky))
        print(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernelsize, stride=stride, padding=padding,
        #                       bias=False)
        # self.init_weight()

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        # x = x.resize_(1, 3, 182, 182)
        return x


input_x = cv2.imread("1.jpg")
cv2.imshow("input_x", input_x)
input_x = cv2.resize(input_x, (368, 368))
input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1).unsqueeze(0)
gaussian_conv = GaussLayer()
print(list(gaussian_conv.parameters()))
out_x = gaussian_conv(input_x)
out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
# out_x = cv2.resize(out_x, (46, 46))
# ret, out_x = cv2.threshold(out_x, 0, 255, cv2.THRESH_BINARY)
cv2.imshow("out_x", out_x)
cv2.waitKey(0)

x = torch.rand(1, 3, 368, 368).float()
input_names = ["inputs"]
output_names = ["outputs"]
# Export the model
torch_out = torch.onnx.export(gaussian_conv, x, "gauss.onnx", export_params=True, verbose=True,
                              input_names=input_names, output_names=output_names)
print("export onnx done")
# learning_rate = 0.01
# input_img = cv2.imread("1.jpg")
# output_img = input_img.copy()
# intput = torch.tensor(input_img).float()
# output = torch.tensor(output_img).float()
#
# net = GaussLayer(intput, output)
# params = list(net.parameters())
# optimizer = torch.optim.Adam(net.parameters())
# loss_function = nn.MSELoss()
#
# for epoch in range(5000):
#     out = GaussLayer(intput, output)
#     loss = loss_function(out, output)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# print(params)
# print(GaussLayer(intput).data)


# cv2.imshow("", output)

# def init_weight(self):
#     for ly in self.children():
#         if isinstance(ly, nn.Conv2d):
#             nn.init.kaiming_normal_(ly.weight, a=1)
#             if not ly.bias is None:
#                 nn.init.constant_(ly.bias, 0)
#
# def gausskernel_2d_opencv(self, kernelsize=3, sigma=0):
#     kx = cv2.getGaussianKernel(kernelsize, sigma)
#     ky = cv2.getGaussianKernel(kernelsize, sigma)
#     return np.multiply(kx, np.transpose(ky))
#
# def gausskernel_2d(self, kernelsize=3, sigma=0):
#     kernel = np.zeros([kernelsize, kernelsize])
#     center = (kernelsize - 1) / 2
#
#     if sigma == 0:
#         sigma = ((kernelsize - 1) * 0.5 - 1) * 0.3 + 0.8
#
#     s = 2 * (sigma ** 2)
#     sum_val = 0
#     for i in range(0, kernelsize):
#         for j in range(0, kernelsize):
#             x = j - center
#             y = j - center
#             kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
#             sum_val += kernel[i, j]
#     sum_val = 1 / sum_val
#     return kernel * sum_val
