import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        '''
            Initializes an instance of this class with default attributes:
            split_size, num_boxes, num_classes, lambda_coord, and lambda_noobj.
        '''

        super(YoloLoss, self).__init__()
        # MSE = Min square error.
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> float:
        '''
            Calculate the total loss of the model in the training process.
            The total loss contains box loss, object loss and class loss.

            Args:
                predictions (torch.Tensor): the prediction after forwarding the
                the input through the entire network. The shape of prediction
                is (batch, S * S * (C + B * 5)).
                target (torch.Tensor): also means the lable, this should be
                in the same shape of predictions before calculating loss.
            
            Returns:
                float: value of the total loss.
        '''

        # predictions.shape = (batch, 7, 7, 30)
        # target.shape = (batch, 7, 7, 25)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # predictions[..., 2:6] is the first predicted bounding box.
        # predictions[..., 7:11] is the second predicted bounding box.

        # target[..., 2:6] is the true bounding box (label).

        # predictions[..., 2:6].shape = (batch, 7, 7, 4)
        # predictions[..., 7:11].shape = (batch, 7, 7, 4)

        # target[..., 2:6].shape = (batch, 7, 7, 4)

        # iou_b1.shape = iou_b2.shape = (batch, 7, 7, 1)
        iou_b1 = intersection_over_union(predictions[..., 2:6], target[..., 2:6])
        iou_b2 = intersection_over_union(predictions[..., 7:11], target[..., 2:6])

        # Add an additional dimension to iou_b1 and iou_b2 to find the
        # highest IOU value among them.

        # iou_b1.unsqueeze(0).shape = (1, batch, 7, 7, 1)
        # iou_b2.unsqueeze(0).shape = (1, batch, 7, 7, 1)

        # torch.cat() is to concatenate two tensor following the
        # additional dimension that has been added.

        # ious.shape = (2, batch, 7, 7, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the higest IOU values and their indices.

        # iou_maxes is a tensor that contains higher IOU values.
        # besbox is a tensor that contains indices of higher IOU values.

        # Note that elements in bestbox are 0 and 1 only.
        # 0 means that box predictions[..., 2:6] has higher IOU.
        # 1 means that box predictions[..., 7:11] has higher IOU.

        # iou_maxes.shape = (batch, 7, 7, 1)
        # bestbox.shape = (batch, 7, 7, 1)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # This is the value of I_{ij}^{obj} in the yolov1 paper.
        # exists_box.shape = (batch, 7, 7, 1)
        exists_box = target[..., 1].unsqueeze(3)

        # ========== # 
        #  BOX LOSS  #
        # ========== #

        # bestbox * predictions[..., 7:11] is to keep second bounding boxes
        # that have higher IOU values than the first ones.

        # bestbox * predictions[..., 2:6] is to keep first bounding boxes
        # that have higher IOU values than the second ones.

        # box_predictions contains all bouding boxes that have
        # higher IOU values.

        # box_predictions.shape = (batch, 7, 7, 4)
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 7:11]
                + (1 - bestbox) * predictions[..., 2:6]
            )
        )

        box_targets = exists_box * target[..., 2:6]

        # Take sqrt of width, height of boxes
        # torch.sign() is to take the sign of numbers in tensor.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ============= #
        #  OBJECT LOSS  #
        # ============= #

        # pred_box is the confidence score for the bbox with highest IOU
        # pred_box.shape = (batch, 7, 7, 1)
        pred_box = (
            bestbox * predictions[..., 6:7] + (1 - bestbox) * predictions[..., 1:2]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 1:2]),
        )

        # ================ #
        #  NO OBJECT LOSS  #
        # ================ #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 1:2], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 6:7], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1)
        )

        # ============ #
        #  CLASS LOSS  #
        # ============ #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :1], end_dim=-2,),
            torch.flatten(exists_box * target[..., :1], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
