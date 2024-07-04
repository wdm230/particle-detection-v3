import torch
from torchvision.transforms import functional as F 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from model import get_model
import glob
import random
import os

def load_and_infer(model, img_path, device):
    """
    Load an image and perform inference using the model.

    Args:
        model (torch.nn.Module): The trained model for inference.
        img_path (str): Path to the input image.
        device (torch.device): The device to run the inference on (CPU or CUDA).

    Returns:
        tuple: The input image and the model's output.
    """
    image = Image.open(img_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    return image, outputs[0]

def visualize_results(filename, image, outputs, threshold=0.5):
    """
    Visualize the results of the model's predictions.

    Args:
        image (PIL.Image.Image): The input image.
        outputs (dict): The model's output containing bounding boxes and scores.
        threshold (float, optional): The score threshold for displaying bounding boxes. Defaults to 0.5.
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for idx, box in enumerate(outputs['boxes']):
        if outputs['scores'][idx] > threshold:
            xmin, ymin, xmax, ymax = box.cpu().numpy()  # Move to CPU and convert to numpy
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        # Extract the base name (file name with extension)
    base_name = os.path.basename(filename)
    # Split the file name and extension
    file_name, _ = os.path.splitext(base_name)
    plt.savefig(f"data/plots/{file_name}")
    plt.show()

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Load the model
num_classes = 2  # 1 class (particle) + background
model = get_model(num_classes)
model.load_state_dict(torch.load('data/saved_model/particle_detector.pth', map_location=device))
model.to(device)

# Get list of image files
imgs = glob.glob("data/img/images/*.png")

# Perform inference and visualize results for each image
for file in imgs:
    image, outputs = load_and_infer(model, file, device)
    visualize_results(file, image, outputs)
