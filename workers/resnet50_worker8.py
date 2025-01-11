# Environment Setup and Library Imports
from torchvision.models import resnet50
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.io import read_image
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt
import copy
from torch.distributed import all_reduce, ReduceOp
from tqdm import tqdm
import warnings
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initializes the custom dataset object.

        Args:
            root_dir (str): The root directory to search for classes.
            transform (callable, optional): A function/transform to apply to the samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imgs = self._make_dataset()

    def _find_classes(self, dir):
        """
        Finds class folders in a directory.

        Args:
            dir (str): Root directory path.

        Returns:
            tuple: A tuple containing:
                - classes (list): List of class names.
                - class_to_idx (dict): Dictionary mapping class names to indices.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        """
        Creates a list of (image_path, class_index) tuples.

        Returns:
            list: List of tuples where each tuple contains the path to an image
                  and its corresponding class index.
        """
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for root, _, fnames in os.walk(class_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        images.append((path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The transformed image.
                - label (int): The class label of the image.
        """
        img_path, label = self.imgs[idx]
        image = read_image(img_path).float() / \
            255.0  # Normalize image to [0, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def setup(rank, world_size):
    """
    Sets up the environment for distributed training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes involved in the training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        """
        Builds a semantic segmentation model using ResNet50 as the backbone.

        Args:
            num_classes (int, optional): Number of segmentation classes. Defaults to 3.
            input_channels (int, optional): Number of input channels. Defaults to 3.
        """
        super(SegmentationModel, self).__init__()

        # Load ResNet50 as the backbone
        base_model = resnet50(pretrained=True)
        # Remove FC and AvgPool layers
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        # Freeze the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Custom upsampling layers
        self.upsample = nn.Sequential(
            # First upsampling block
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # 2048 -> 512
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 16x16 -> 32x32

            # Second upsampling block
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 32x32 -> 64x64

            # Third upsampling block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 64x64 -> 128x128

            # Fourth upsampling block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 128x128 -> 256x256

            # Final layer to predict segmentation classes
            nn.Conv2d(64, num_classes, kernel_size=1)  # 64 -> num_classes
        )

    def forward(self, x):
        """
        Forward pass for the segmentation model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, H, W).
        """
        x = self.backbone(x)
        x = self.upsample(x)
        return x


def initial_model(input_shape=(256, 256, 3), num_classes=3):
    """
    Constructs the segmentation model.

    Args:
        input_shape (tuple, optional): Shape of input images (H, W, C). Defaults to (256, 256, 3).
        num_classes (int, optional): Number of segmentation classes. Defaults to 3.

    Returns:
        nn.Module: The segmentation model.
    """
    input_channels = input_shape[2]
    model = SegmentationModel(num_classes=num_classes,
                              input_channels=input_channels)
    return model


def cleanup():
    """
    Cleans up the distributed training environment.
    """
    dist.destroy_process_group()


# ----- DataLoader with DistributedSampler -----
# Prepares PyTorch DataLoader with a DistributedSampler to efficiently distribute the data across multiple GPUs with 8 workers.
def prepare_dataloader(dataset, batch_size, world_size, rank, num_workers=8):
    """
    Prepares a DataLoader with a DistributedSampler for distributed training.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        world_size (int): The total number of processes in the training.
        rank (int): The rank of the current process.
        num_workers (int, optional): The number of subprocesses to use for data loading.

    Returns:
        DataLoader: The DataLoader configured for distributed training.
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader


def get_optimizer_and_criterion(model, learning_rate=0.001):
    """
    Returns an optimizer for the given model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def train_resnet_model(
    model, criterion, optimizer, data_loaders, rank, world_size, num_epochs=3
):
    """
    Train a model using distributed training.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        data_loaders (dict): Dictionary containing 'train' and 'val' DataLoader objects.
        rank (int): The rank of the current process.
        world_size (int): Total number of processes participating in the training.
        num_epochs (int, optional): Number of epochs to train. Defaults to 3.

    Returns:
        dict: A dictionary containing training metrics like loss, accuracy, and time.
    """

    start_time = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "time": []
    }

    try:
        for epoch in range(num_epochs):
            print(f"Rank {rank}: Epoch {epoch + 1}/{num_epochs}")

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                loader = data_loaders[phase]
                for inputs, labels in tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch + 1}"):
                    inputs = inputs.to(rank)
                    labels = labels.to(rank)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    total_samples += inputs.size(0)

                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects / total_samples

                # Save metrics
                if phase == 'train':
                    metrics["train_loss"].append(epoch_loss)
                    metrics["train_acc"].append(epoch_acc)
                else:
                    metrics["val_loss"].append(epoch_loss)
                    metrics["val_acc"].append(epoch_acc)

                print(
                    f"Rank {rank} - {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
                )

                # Update the best model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        total_time = time.time() - start_time
        metrics["time"].append(total_time)
        print(f"Rank {rank}: Training completed in {total_time:.2f} seconds")

        # Load the best model weights
        model.load_state_dict(best_model_wts)

    except Exception as e:
        print(f"Rank {rank}: Exception during training: {e}")
        raise e

    finally:
        cleanup()

    return metrics


def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on a test set and saves predictions and true labels for analysis.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to perform evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing:
            - 'accuracy': The overall accuracy of the model on the test set.
            - 'predictions': A list of predicted labels.
            - 'true_labels': A list of ground truth labels.
            - 'confusion_matrix': Confusion matrix for the test set.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Model"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

            total_correct += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)

    # Calculate overall accuracy
    accuracy = total_correct / total_samples

    # Compute confusion matrix
    confusion_mat = confusion_matrix(true_labels, predictions)

    # Log results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion_mat}")

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "true_labels": true_labels,
        "confusion_matrix": confusion_mat,
    }


def main(rank, world_size, num_epochs, batch_size, train_dir, val_dir, test_dir):
    """
    Main function for distributed training.

    Arguments:
        rank (int): The unique identifier for the current process in the distributed setup.
                    Each process corresponds to one GPU and has a rank ranging from 0 to world_size-1.
                    Rank 0 is typically the main process responsible for logging and saving models.
        world_size (int): The total number of processes participating in the distributed setup.
                          This is usually equal to the number of GPUs available for training.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The global batch size, which is divided equally across all processes.
        train_dir (str): The directory containing the training dataset.
        val_dir (str): The directory containing the validation dataset.
        test_dir (str): The directory containing the test dataset.

    Explanation of rank and world_size in this context:
        - `rank`: Identifies which process is running. For example:
            - Rank 0 handles the main coordination tasks like logging and saving checkpoints.
            - Other ranks are responsible for processing their respective data shard.
        - `world_size`: The total number of processes (or GPUs) involved in training.
            It determines how data and workload are split across processes.

    Example:
        If you are using 4 GPUs for distributed training:
        - `world_size` = 4
        - `rank` ranges from 0 to 3, each corresponding to a specific GPU.

    Workflow:
        - Each process (identified by its `rank`) works on a portion of the dataset.
        - All processes synchronize their results during backpropagation using distributed communication.
    """
    try:
        setup(rank, world_size)

        train_dataset = CustomDataset(root_dir=train_dir)
        val_dataset = CustomDataset(root_dir=val_dir)
        train_loader = prepare_dataloader(
            train_dataset, batch_size, world_size, rank)
        val_loader = prepare_dataloader(
            val_dataset, batch_size, world_size, rank)
        test_dataset = CustomDataset(root_dir=train_dir)
        test_loader = train_loader = prepare_dataloader(
            test_dataset, batch_size, world_size, rank)

        model = initial_model()  # initialize the model here via function
        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False)

        optimizer, criterion = get_optimizer_and_criterion(model)

        data_loaders = {"train": train_loader, "val": val_loader}

        metrics = train_resnet_model(
            model, criterion, optimizer, data_loaders, rank, world_size, num_epochs
        )

        if rank == 0 and world_size == 1:
            model.load_state_dict(
                torch.load(
                    f"resnet50_8W/resnet_model_{world_size}g_8w.pth")
            )
            evaluate_model(model, test_loader, rank)

    except Exception as e:
        print(f"Error in process {rank}: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    """
    Maximum GPUs we will be using is 4
    Num of Epochs = 5
    Batch size = 128 ( This will be run and results are verified if out of memory batch size to be reduced 
    if running into out of memory batch size to be reduced )
    """
    max_gpus = 4
    num_epochs = 5
    batch_size = 128
    train_dir = "/home/gaali.v/Final-Project/data/train"
    val_dir = "/home/gaali.v/Final-Project/data/val"
    test_dir = "/home/gaali.v/Final-Project/data/test"
    for world_size in range(1, max_gpus + 1):
        print(f"Training with {world_size} GPU(s)")

        mp.spawn(
            main,
            args=(world_size, num_epochs, batch_size,
                  train_dir, val_dir, test_dir),
            nprocs=world_size,
            join=True,
        )

        print(f"Completed training with {world_size} GPU(s)\n")
