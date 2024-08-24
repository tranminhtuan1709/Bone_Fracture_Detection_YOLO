import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file: str, img_dir: str, label_dir: str,
        S=7, B=2, C=1,
        transform=None,
    ) -> None:
        '''
            Initializes an instance of this class.

            Args:
                csv_file (str): path to the csv file.
                img_dir (str): path to the image folder.
                label_dir (str): path to the label folder.
                S (int): default = 7
                B (int): default = 2
                C (int): default = 20
                tranform (Any): default = None

            Returns:
                None
        '''

        self.annotations = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C


    def __len__(self) -> int:
        '''
            Returns the number of training examples in the training set.

            Args:
                None

            Returns:
                int: the number of training examples.
        '''

        return len(self.annotations)


    def __getitem__(self, index: int) -> tuple:
        '''
            Get the image and label matrix of a training example
            at the given index.

            Args:
                index (int): index of the required tranining example.

            Returns:
                tuple: a tuple (PIL.Image, torch.Tensor) contains an image and
                a label matrix.
        '''

        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        # Read data in the label file.
        with open(label_path) as f:
            for label in f.readlines():
                if label.strip():
                    class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                    ]

                boxes.append([class_label, x, y, width, height])

        # Read image.
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert the list of bounding boxes into a tensor.
        # The shape of the label need to be the same as the output
        # of the model. Som it need to be (S, S, B * 5 + C).

        # For each bounding box in the label file:
        # + Find the cell that contains the center point.
        # + Finds relative coordinates in a single cell.
        # + Find relative width and height.
        # + Add bounding box and its class to the label matrix.

        # Note that cell does not contains any bounding box will be full of 0.
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i, j represents the position (row, column) of the cell that contains
            # the center point of the bounding box.
            i, j = int(self.S * y), int(self.S * x)

            # Relative coordinates of the center point in a cell.
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # Relative width and height of the bounding box.
            width_cell, height_cell = width * self.S, height * self.S

            # Check if the cell already contains a bounding box.
            # This means there is only one bounding box in a cell.
            if label_matrix[i, j, 1] == 0:
                # Set that there exists an object.
                label_matrix[i, j, 1] = 1

                # Box coordinates.
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 2:6] = box_coordinates

                # Set one hot encoding for class_label.
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
