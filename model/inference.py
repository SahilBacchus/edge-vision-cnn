import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import EdgeCNN


# CIFAR-10 class names 
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]



def get_transform():
    '''
    Get transforms needed on image for inference
    '''
    # Mean & Std of CIFAR10 dataset --> precomputed from online
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    inference_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    return inference_transform


@torch.no_grad()
def load_model(model_path, device="cpu"):
    '''
    Load EdgeCNN model weights from path
    '''
    model = EdgeCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def predict(model, image):
    '''
    Get predicted label for image with confidence
    '''
    # Preprocess image 
    transform = get_transform()
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    # Forward pass with image --> predict
    logits = model(image)
    probs = F.softmax(logits, dim=1).squeeze(0)  # Converts the outputs of last linear layer (logits) to confidence/probabilties for each class

    # Get top prediction with confidence
    confidence, pred_idx = torch.max(probs, dim=0)
    pred_class = CLASS_NAMES[pred_idx.item()]

    return pred_class, confidence.item()



