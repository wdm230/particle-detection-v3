import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_model(num_classes):
    """
    Create a Faster R-CNN model with a ResNet-50 FPN backbone, pre-trained on COCO dataset.

    Args:
        num_classes (int): Number of classes for the classifier (including background).

    Returns:
        model (torchvision.models.detection.FasterRCNN): Faster R-CNN model with the specified number of classes.
    """
    # Use the weights parameter instead of pretrained
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
