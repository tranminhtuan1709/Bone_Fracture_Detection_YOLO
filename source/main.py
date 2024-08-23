import model
import loss
import train
import utils
import dataset
import torch
import matplotlib.pyplot as plt
from PIL import Image


yolo = model.Yolov1(split_size=7, num_boxes=2, num_classes=20)

# x.shape = (1, 3, 448, 448)
x = torch.randn(1, 3, 448, 448)

# output.shape = (1, 1470)
output = yolo(x)

# output.shape = (1, 7, 7, 30)
output = output.reshape(-1, 7, 7, 30)

yololoss = loss.YoloLoss()

# label.shape = (1, 7, 7, 30)
label = torch.zeros(1, 7, 7, 30)

label[0, 0, 0, 10] = 1
label[0, 0, 0, 20] = 1
label[0, 0, 0, 21] = 0.34419263456090654
label[0, 0, 0, 22] = 0.611
label[0, 0, 0, 23] = 0.4164305949008499
label[0, 0, 0, 24] = 0.262

yololoss(output, label)
