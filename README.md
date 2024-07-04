
# Particle Detection with Faster R-CNN

This repository contains the code for training and evaluating a Faster R-CNN model for particle detection on images of nuclepore filters. The project uses PyTorch and torchvision for implementing the object detection model.

## Project Structure

```
particle-detection/
│
├── dataset.py        # Custom dataset class for loading images and annotations
├── model.py          # Function to create the Faster R-CNN model
├── train.py          # Script for training the model with early stopping
├── test_model.py     # Script for testing the model on sample images
├── data/
│   ├── img/
│   │   ├── images/       # Directory containing images
│   │   └── Annotations/  # Directory containing Pascal VOC XML annotations
│   └── saved_model/
│       └── best_particle_detector.pth  # Directory to save the best model
└── README.md          # Project documentation
```

## Requirements

- Python 3.8 or later
- PyTorch
- torchvision
- matplotlib
- Pillow


## Dataset Preparation

Place your images in the `data/img/images/` directory and the corresponding Pascal VOC XML annotations in the `data/img/Annotations/` directory. Ensure that the image and annotation filenames match.

## Training the Model

You can train the model using the `train.py` script.


python train.py


The best model will be saved in the `data/saved_model/` directory as `particle_detector.pth`.

## Testing the Model

To test the model on a random image from the dataset, use the `test_model.py` script:

python test_model.py


This script will load a random image, perform inference using the trained model, and visualize the results with both ground truth and predicted bounding boxes.

## Customizing the Model

The `model.py` script defines the `get_model` function to create the Faster R-CNN model. You can customize this function to change the model architecture or hyperparameters.

## Issues and Contributions

If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

