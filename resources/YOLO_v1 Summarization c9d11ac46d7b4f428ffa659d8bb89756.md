# YOLO_v1 Summarization

Paper : https://arxiv.org/pdf/1506.02640

# 1. Introduction

YOLO_v1 is a new approach to object detection, it frames object detection as a single regression problem, straight from original images to coordinates of bounding boxes and respective class probabilities.

YOLO_v1 is designed as a unified system, which has several benefits compared to traditional ones :

- YOLO is extremely fast.
- YOLO reasons globally about the image when making predictions.
- YOLO learns generalizable representations of objects.

# 2. Unified Detection

> Terminology
> 
- ***Bounding boxes*** is a rectangle that frames around objects. Those are represented by five scalars $x,\space y,\space w,\space h$ where :
    - $(x, y)$ is the coordinates of the center of a bounding box.
    - $(w, h)$ is the weight and height of a bounding box respectively.
- ***Confidence score***  is a value that reflects the probability of a bounding box to contain an object and how well it fits that object. Confidence score is represented by :

$$
Pr(Object)*IOU_{pred}^{truth}
$$

YOLO explicitly points out that the confidence score should be 0 if there is no object in the cell. Adversely, it should be 1. This implicitly means that :

$$
\begin{cases}
Pr(Object)=0\quad,\ \textsf{if no object in the cell} \\
Pr(Object)=1\quad,\ \textsf{in the opposite case}
\end{cases}
$$

> General technique
> 
- The original image is divided into an $S \times S$ grid.
- Each grid cell is responsible for detecting B bounding boxes with their respective confidence scores and $C$ conditional class probabilities.
- A bounding box has the highest confidence score is used as the localization result while others are suppressed.
- To represent the final result of a bounding box, YOLO multiplies the confidence score with the conditional class probability :

$$
Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}=Pr(Class_i)*IOU_{pred}^{truth}
$$

## 2.1. Network Design

Implement : uses a convolutional neural network.

Evaluate : Pascal VOC detection dataset.

![YOLO_v1 network](YOLO_v1%20Summarization%20c9d11ac46d7b4f428ffa659d8bb89756/image.png)

YOLO_v1 network

Details about network :

- Inspired by GooLeNet.
- 24 convolutional layers.
- 2 fully connected layers.
- Uses $1\times 1$ reduction layers.

A smaller version is Fast YOLO with 9 layers and the same parameters.

## 2.2. Training

YOLO_v1 is pre-trained on the ImageNet 1000-class competition dataset using Darknet framework.

YOLO_v1 normalizes the bounding box width and height to be relative with a cell ($x,\ y,\ w,\ h \in[0,1]$)

A linear activation function is used for the final layer and all others - Leaky ReLU

$$
\phi(x)=\begin{cases}
x\quad\quad\ ,\ \textsf{if}\ x>0 \\
0.1x\quad,\ \textsf{otherwise}
\end{cases}
$$

Loss function used to train YOLO_v1 has three main part :

> Location loss
> 

$$
\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}\Big[(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2\Big] \\
+\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}\Bigg[\Big(\sqrt{w_i}-\sqrt{\hat{w}_i}\Big)^2+\bigg(\sqrt{h_i}-\sqrt{\hat{h}_i}\bigg)^2\Bigg]
$$

Where :

- $1_{ij}^{obj}=1$ if cell $i$ contains the object, and $1_{ij}^{obj}=0$ in the opposite case.
- $\hat{x}_i,\ \hat{y}_i$ : the center of the truth object in cell $i$.
- $x_i,\ y_i$ : the center of the bounding box predicted by model in cell $i$.
- $\hat{w}_i,\ \hat{h}_i$ : weight and height of the truth object in cell $i$.
- $w_i,\ h_i$ : weight and height of the bounding box predicted by model in cell $i$.
- $\lambda_{coord}=5$ is used to increase the loss from bounding box coordinate predictions.

! Square roots of the bounding box width and height are used to increase the loss from smaller bounding boxes.

> Object loss
> 

$$
\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}\Big(C_i-\hat{C}_i\Big)^2+\lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{noobj}\Big(C_i-\hat{C}_i\Big)^2
$$

Where :

- $\hat{C}_i=Pr(Object)$ of bounding box $j$ of cell $i$.
- $C_i=1$ if cell $i$ contains the object, and $C_i=0$ in the opposite case.
- $\lambda_{noobj}=0.5$ to decrease the loss from confidence predictions for boxes that do not contain objects.

> Classification loss
> 

$$
\sum_{i=0}^{S^2}1_{i}^{obj}\sum_{c\in class}^{}\big(p_i(c)-\hat{p}_i(c)\big)^2
$$

Where :

- $\hat{p}_i(c)=1$ if cell $i$ contains an object of class $c$, and $\hat{p}_i(c)=0$ in the opposite case.
- $p_i(c)=Pr(Class_c|Object)$.

> Total loss
> 

$$
\boxed{\textsf{Total loss = Location loss + Object loss + Classification loss}}
$$

> Training parameters
> 
- Epochs = 135.
- Batch size = 64.
- Momentum = 0.9.
- Decay = 0.0005.
- Learning rate :
    - First epochs = $10^{-3}\rightarrow 10^{-2}$.
    - 75 epochs = $10^{-2}$.
    - 30 epochs = $10^{-3}$.
    - 30 epochs = $10^{-4}$.
- Dropout = 0.5
- Augmentation = [scaling, translation, exposure, saturation]

## 2.3. Inference

- Predicting detections for a test image only requires one network evaluation.
- Non-max suppression is used to fix the multiple detections problem.

## 2.4. Limitation of YOLO

- The spatial constraint limits the number of nearby objects that can be predicted.
- Struggles to generalize to objects in new or unusual aspect ratios or configurations.
- Struggles in detecting small objects that appear in groups.
- Loss function treats errors the same in small bounding boxes and the large ones.
- The main source of error is incorrect localizations

# 3. Comparison to Other Detection Systems

- DPM → Sliding window.
- R-CNN → Region proposal.
- YOLO → Grid approach based on MultiGrasp system for regression to grasps.

# 4. Experiments

## 4.1. Comparison to Other Real-Time Systems

- Fast YOLO is the fastest object detection method on PASCAL with 52.7% mAP.
- YOLO’s mAP is 63.4% while still maintaining real-time performance.
- Fastest DPM misses real-time performance although speeds up mAP.
- Fast R-CNN relies on so it has high mAP but very slow speed.
- Faster R-CNN is the most accurate model but it just achieves 7 fps while a smaller, less accurate one runs at 18 fps.

## 4.2. VOC 2007 Error Analysis

Comparison between YOLO and Fast R-CNN:

- YOLO struggles in localizing objects correctly.
- Fast R-CNN makes much fewer localization errors but more background errors.

## 4.3. Combining Fast R-CNN and YOLO

Because of the higher accuracy in detecting background than Fast R-CNN, it can be used to diminish errors in background detections from Fast R-CNN.

→ For every bounding box that R-CNN predicts, we check whether YOLO will predict the same one.

⇒ mAP of combined model (YOLO and Fast R-CNN) is higher than the single ones.

However, this combination does not benefit YOLO in terms of speed, but YOLO is fast so it does not affect computational speed when combining to Fast R-CNN.

## 4.4. VOC 2012 Results

On the VOC 2012 test set :

- YOLO scores 57/9%.
- Combined Fast R-CNN + YOLO model is one of the highest performing detection methods.

## 4.5. Generalizability : Person Detection in Artwork

Comparison between models in Artwork detection :

- R-CNN has high AP on detecting people, but lower in analyzing artwork.
- DPM has stable AP when applied to artwork.
- YOLO has good performance and its AP degrades less than other ones when applied to artwork.

# 5. Real-Time Detection In the Wild

YOLO is a fast and accurate detection model, hence, it can be implemented in computer vision applications like webcam, etc.

# 6. Conclusion

- YOLO is a unified detection model implemented using a convolutional neural network and use sum-squared loss function.
- YOLO is fast.
- YOLO has high accurate even when applied to artwork.
- YOLO also has some drawbacks.
- YOLO in comparison with other state-of-the-art models.