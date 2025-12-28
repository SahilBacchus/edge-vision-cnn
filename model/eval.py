import torch
import time
import os
from collections import OrderedDict

from data import get_dataloaders
from model import EdgeCNN




# ========================= #
# Define metric calculations
# ========================= #

@torch.no_grad()
def evaluate_accuracy(model, dataloader):
    '''
    Evaluate model accuracy on data (correct / total * 100)
    '''
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100.0


def count_parameters(model):
    '''
    Count the total number of parameters in model (trainable + non-trainable)
    '''
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """
    Calculate model size in MB (weights + buffers)
    """
    total_bytes = 0

    # Add all parameters
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()

    # Add all buffers
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()

    return total_bytes / (1024**2)


@torch.no_grad()
def measure_latency(model, input_size=(1, 3, 32, 32), warmup=50, runs=200):
    '''
    Measure inference time on dummy data
    '''
    model.eval()
    dummy = torch.randn(input_size)

    # Warm‑up
    _ = [model(dummy) for _ in range(warmup)]

    # Time model inference
    start = time.time()
    _ = [model(dummy) for _ in range(runs)]
    end = time.time()

    latency_ms = (end - start) / runs * 1000
    return latency_ms


def estimate_inference_cost(latency_ms, cpu_cost_per_hour=0.05, num_inferences=1000):
    """
    Estimate cost per num_inferences
    """
    latency_s = latency_ms / 1000.0
    inferences_per_hour = 3600 / latency_s
    cost = cpu_cost_per_hour * (num_inferences / inferences_per_hour)

    return cost


def load_models():
    '''
    Load our model along with a set of pre-defined models for evaluation
    '''
    models = OrderedDict()

    # Load EdgeCNN
    edgecnn = EdgeCNN()
    edgecnn.load_state_dict(torch.load("edgecnn_v1.2.pth", map_location="cpu"))
    models["EdgeCNN (Ours)"] = edgecnn

    # Download/load CIFAR‑10 pretrained models
    # GitHub repo w/ models: https://github.com/chenyaofo/pytorch-cifar-models
    models["ResNet20"] = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True
    )
    models["MobileNetV2_x0_5"] = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_mobilenetv2_x0_5",
        pretrained=True
    )
    models["ResNet56"] = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet56",
        pretrained=True
    )
    models["MobileNetV2_x1_4"] = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_mobilenetv2_x1_4",
        pretrained=True
    )

    # Set all to eval mode
    for m in models.values():
        m.eval()

    return models




# ========================= #
# Main eval script
# ========================= #

def main():

    print("=" * 90)
    print(" Model Evaluation ".center(90, "="))
    print("=" * 90)

    device = torch.device("cpu")  # Force CPU-only
    print(f"\nRunning on: {device}")

    # Load CIFAR-10 test set
    print("\nLoading data... ")
    _, test_loader = get_dataloaders(batch_size=128)

    # Load our model along with others to test
    print("\nLoading models... ")
    models = load_models()


    # Calculate metrics for each model
    print()
    print(" Model Comparison on CIFAR-10 ".center(90, "-"))
    print(
        f"{'Model':20s} | {'Acc (%)':8s} | {'Params (M)':10s} | "
        f"{'Size (MB)':10s} | {'Latency (ms)':12s} | "
        f"{'$/1k Inferences':15s}"
    )
    print("-" * 90)

    for name, model in models.items():
        acc = evaluate_accuracy(model, test_loader)
        params = count_parameters(model)
        size_mb = model_size_mb(model)
        latency = measure_latency(model)
        cost_per_1k = estimate_inference_cost(latency, cpu_cost_per_hour=0.05, num_inferences=1000)

        print(
            f"{name:20s} | "
            f"{acc:8.2f} | "
            f"{params/1e6:10.3f} | "
            f"{size_mb:10.3f} | "
            f"{latency:12.3f} | "
            f"{cost_per_1k:15.6f}"
        )


if __name__ == "__main__":
    main()

