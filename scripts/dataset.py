import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ParticlesDataset(Dataset):
    """
    Custom dataset for loading images and annotations in Pascal VOC format.

    Args:
        root_dir (str): Root directory containing images and annotations.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'images')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.annotation_files = [os.path.splitext(img)[0] + '.xml' for img in self.image_files]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, target) where image is a transformed image tensor and
                   target is a dictionary containing bounding boxes and labels.
        """
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        ann_path = os.path.join(self.ann_dir, self.annotation_files[idx])

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_voc_xml(ann_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target

    def parse_voc_xml(self, filename):
        """
        Parses a Pascal VOC XML file and extracts bounding boxes and labels.

        Args:
            filename (str): Path to the XML file to be parsed.

        Returns:
            tuple: (boxes, labels) where boxes is a list of bounding boxes and
                   labels is a list of class labels.
        """
        tree = ET.parse(filename)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming one class

        return boxes, labels

# Transformation to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.

    Args:
        batch (list): List of tuples where each tuple contains an image and a target.

    Returns:
        tuple: (images, targets) where images is a list of image tensors and
               targets is a list of dictionaries containing bounding boxes and labels.
    """
    return tuple(zip(*batch))
