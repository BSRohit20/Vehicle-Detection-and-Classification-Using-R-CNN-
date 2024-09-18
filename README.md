# VEHICLE DETECTION AND CLASSIFICATION-USING-AND-R-CNN
This repository contains the implementation of an object detection system using Faster R-CNN (Region-based Convolutional Neural Networks) with the ResNet-50 backbone. The system is designed to detect objects (specifically cars in this case) in images and return bounding boxes along with classification labels. The code supports training, validation, and testing on a custom dataset.

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Training](#training)
6. [Testing](#testing)
7. [Results](#results)

## Project Description
This project implements a car detection system using PyTorch's Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Networks (FPN). The model is fine-tuned on a custom dataset that contains bounding boxes for cars in images. The project includes data preprocessing, model training, validation, and testing steps.

## Installation
To run the code, you need Python 3.7+ and the following libraries:

```bash
pip install -qU torch torchvision torch_snippets scikit-learn pandas Pillow opencv-python
```

Ensure that you have a CUDA-enabled GPU for faster model training, though the code can also run on a CPU (with significantly slower training).

## Dataset
The dataset contains images with bounding boxes for cars. The bounding box annotations are provided in a CSV file (`train_solution_bounding_boxes.csv`), which has columns for the image ID and the bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`). The images are stored in the `data/training_images` directory.

To prepare the dataset:
1. Place the images in the `data/training_images` folder.
2. The bounding box annotations should be in a CSV file located at `data/train_solution_bounding_boxes.csv`.

## Model
We use a Faster R-CNN model pre-trained on COCO and fine-tune it for our car detection task. The ResNet-50 backbone is used, and the number of output classes is adjusted to match the custom dataset's labels.

### Model Structure
- **Backbone:** ResNet-50 with Feature Pyramid Networks (FPN).
- **Object Detector:** Faster R-CNN.
- **Optimizer:** SGD with momentum and weight decay.

## Training
The training code is set up to:
1. Load the dataset and split it into training and validation sets (90% training, 10% validation).
2. Train the Faster R-CNN model with Stochastic Gradient Descent (SGD) for 10 epochs.
3. Log the training and validation losses, including classification loss, box regression loss, and objectness loss.

### Running Training
```python
python train.py
```

The training script will:
- Load the dataset from the `data/training_images` folder.
- Train the model using the provided images and bounding box annotations.
- Validate the model after each epoch.

## Testing
After training, you can test the model on a batch of images by running the following:

```python
python test.py
```

The test script will:
- Load the trained model.
- Pass test images through the model.
- Display the bounding boxes along with confidence scores and labels for each detected object.

## Results
During inference, the system will display the images with detected bounding boxes and labels. Non-Maximum Suppression (NMS) is applied to filter overlapping boxes based on Intersection over Union (IoU). The bounding boxes, confidence scores, and labels are decoded and visualized.

## Example Output
For each detected object, the bounding box and class label will be displayed on the image along with the confidence score.

---
