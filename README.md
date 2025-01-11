# **Parallel Deep Learning for Liver Tumor Segmentation**

---

## **Project Overview**

Liver cancer is one of the deadliest forms of cancer worldwide, making accurate tumor segmentation a critical task in diagnosis and treatment planning. This project focuses on leveraging parallel deep learning techniques to enhance the computational efficiency of liver tumor segmentation.

By employing a distributed training setup using multiple GPUs, the project aims to achieve faster model training and inference while maintaining high accuracy and robustness. This enables the processing of large and complex medical datasets efficiently.

---

## **Methodology**

### **Model Architecture**

- **U-Net**: A convolutional neural network architecture designed specifically for biomedical image segmentation.

### **Parallelization**

- Utilized **Distributed Data Parallel (DDP)** in PyTorch to train the model across four GPUs.
- Each GPU processes unique data batches using PyTorch’s `DistributedSampler` for efficient training.

### **Optimization and Loss**

- **Optimizer**: Adam optimizer for effective gradient updates.
- **Loss Function**: Cross-entropy loss to evaluate segmentation performance.

### **Training Setup**

- **Batch Size**: 32
- **Epochs**: 40
- **Framework**: PyTorch for parallel training and efficient gradient computation.

---

## **Dataset**

The dataset consists of liver CT images with corresponding segmentation labels. Each label identifies:

- **Liver regions**: Threshold set to 40.
- **Tumor regions**: Threshold set to 230.

Data is divided into training and validation sets to ensure effective model training and evaluation.

---

## **Results**

### **Key Metrics**

- **Training Loss vs. Epochs**: Steady decrease across all GPUs, indicating convergence.
- **Validation Loss vs. Epochs**: Significant reduction, demonstrating model generalization.
- **Validation Accuracy**: Consistently high values across all GPUs, showcasing effective learning.

### **Performance Highlights**

- Faster training time due to distributed processing.
- Robust segmentation performance, with high accuracy in detecting both liver tissue and tumors.

---

## **Key Insights**

1. **Parallel computing** significantly reduces computational time and enhances scalability for large medical datasets.
2. The **U-Net architecture**, combined with distributed training, achieves reliable and efficient segmentation results.

---

## **Future Work**

- Expand the dataset to improve generalization and robustness.
- Fine-tune hyperparameters to optimize performance further.
- Explore alternative deep learning architectures for enhanced segmentation accuracy.

---

## **References**

- PyTorch Distributed Data Parallel (DDP) Documentation
- U-Net Architecture Paper
- Cross-Entropy Loss and Optimization Techniques

