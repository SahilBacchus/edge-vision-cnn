# edge-vision-cnn

## End-to-End Machine Learning Pipeline
 
EdgeCNN is a fully end-to-end machine learning project, covering the complete lifecycle from design and training to production-ready deployment

**1. Model Architecture**
-  A custom edge-optimized convolutional neural network (CNN)
- Implements MobileNet-style depthwise separable convolution blocks for efficiency 

**2. Training**
- Data Augmentation for better generalization
- Optimized with AdamW and cosine annealing learning rate scheduling
- Best model checkpointing

**3. Evaluation & Benchmarking**
- Quantitative comparison against established CIFAR-10 models 

**4. Production API**
- A FastAPI service for model inference
- Image upload endpoint with class prediction with confidence

**5. Deployment**
- Containerized the application with Docker
- Designed for easy deployment to cloud providers (ex. pulling the image on an AWS EC2 instance)

## Evaluation Results 

Benchmarked locally on CPU, assuming $0.05 / hour computer cost: 
 

| Model            | Acc (%)   | Params (M) | Size (MB) | Latency (ms) | $/1k Inferences |
| ---------------- | --------- | ---------- | --------- | ------------ | --------------- |
| EdgeCNN (Ours)   | 91.70     | **0.136**  | **0.524** | **1.424**    | **0.000020**    |
| ResNet20         | 92.12     | 0.272      | 1.046     | 1.813        | 0.000025        |
| MobileNetV2_x0_5 | 92.41     | 0.700      | 2.743     | 6.113        | 0.000085        |
| ResNet56         | **94.25** | 0.856      | 3.281     | 4.458        | 0.000062        |
| MobileNetV2_x1_4 | 93.88     | 4.334      | 16.715    | 8.625        | 0.000120        |

> **Takeaway:**   
> EdgeCNN achieves competitive accuracy while dramatically reducing inference time, memory footprint, and operational cost
