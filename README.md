# edge-vision-cnn





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
