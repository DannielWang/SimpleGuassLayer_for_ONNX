import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        # x.permute(3, 2, 1, 0)
        return x


input_x = cv2.imread("1.jpg")
cv2.imshow("input_x", input_x)
input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1)
gaussian_conv = GaussianBlurConv()
out_x = gaussian_conv(input_x)
out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
cv2.imshow("out_x", out_x)
cv2.waitKey(0)

x = torch.rand(3, 368, 368).float()
input_names = ["inputs"]
output_names = ["outputs"]
# Export the model
torch_out = torch.onnx.export(gaussian_conv, x, "gausscustom.onnx", export_params=True, verbose=True,
                              input_names=input_names, output_names=output_names)
print("export onnx done")