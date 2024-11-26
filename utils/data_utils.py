# flower
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Merger
from matplotlib import pyplot as plt
# torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch

# utils
import pandas as pd
import numpy as np

# huggingface dataset
from datasets import Dataset
from datasets import concatenate_datasets


def load_dataset(args, partition_id: int):
    # prepare Dataset
    if args.data == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./datasets/mnist/', train=True, download=False)
        testset = torchvision.datasets.MNIST(root='./datasets/mnist/', train=False, download=False)
    elif args.data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=False)
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=False)
    elif args.data == 'emnist':
        trainset = torchvision.datasets.EMNIST(root='./datasets/emnist/', train=True, download=False)
        testset = torchvision.datasets.EMNIST(root='./datasets/emnist/', train=False, download=False)
    else:
        raise ValueError("Invalid dataset")


    # create partitioner
    partitioner = IidPartitioner(
                            num_partitions=args.clients,
                            #partition_by="label",
                        ) if args.isiid else DirichletPartitioner(
                            num_partitions=args.clients,
                            partition_by="label",
                            alpha=args.alpha,
                            seed=args.seed,
                            min_partition_size=500,
                            self_balancing=True,
                            #num_classes_per_partition = 4
                        )
    
    if args.fedproto:
        # combine train and test datasets
        # partition the data to the amount of number of clients
        # create test and train set for each client from the partition.
        # prepare the data loader.

        # combine everything together
        ds_dataset = Dataset.from_dict(convert_to_hf_dataset(trainset,testset))
        
        partitioner.dataset = ds_dataset
        
        # save label distribution
        if args.plot_label_dist:
            save_label_distribution(args, partitioner)
        # load specific client data
        client_data = partitioner.load_partition(partition_id=partition_id)

        # create train and test set for the client
        partition_train_test = client_data.train_test_split(test_size=args.split, seed=args.seed)
        client_train = partition_train_test["train"]
        client_test = partition_train_test["test"]

    else:
        # partition the train set of the dataset
        # use that partition to create trainset 
        # and global test set for each client test set.
        ds_trainset = Dataset.from_dict(convert_to_hf_dataset(trainset))
        ds_testset = Dataset.from_dict(convert_to_hf_dataset(testset))

        # partition the data
        partitioner.dataset = ds_trainset

        # save label distribution
        save_label_distribution(args, partitioner)

        client_train = partitioner.load_partition(partition_id=partition_id)
        client_test = ds_testset    
    
    # Apply the transforms
    if args.data=='mnist':
        _transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif args.data=='cifar10':
        _transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError("Invalid dataset")
    
    def apply_transforms(batch):
        batch["image"] = [_transforms(img) for img in batch["image"]]
        return batch
    
    client_train = client_train.with_transform(apply_transforms)
    client_test = client_test.with_transform(apply_transforms)

    trainloader = DataLoader(client_train, batch_size=args.batchsize)
    testloader = DataLoader(client_test, batch_size=args.batchsize)

    return trainloader, testloader




# def load_mnist_partition(args, partition_id: int):

#     # If fedproto; combine the test and train set and split
#     merger = Merger(
#             merge_config={
#                 "train": ("train", "test"),
#                 })
    
#     #fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
#     fds = FederatedDataset(
#         dataset="ylecun/mnist",
#         preprocessor= merger if args.ufedproto else None,
#         partitioners={
#             "train": IidPartitioner(
#                         num_partitions=args.clients,
#                         #partition_by="label",
#                     ) if args.isiid else DirichletPartitioner(
#                         num_partitions=args.clients,
#                         partition_by="label",
#                         alpha=args.alpha,
#                         seed=args.seed,
#                         min_partition_size=500,
#                         self_balancing=True,
#                         #num_classes_per_partition = 4
#                     ),
#         },
#     )

#     client_train = fds.load_partition(partition_id, split="train")
#     partition_train_test = client_train.train_test_split(test_size=args.split, seed=args.seed)
    
#     # fig, ax, df = plot_label_distributions(
#     #                     fds.partitioners["train"],
#     #                     label_name="label",
#     #                     plot_type="bar",
#     #                     size_unit="absolute",
#     #                     partition_id_axis="x",
#     #                     legend=True,
#     #                     verbose_labels=True,
#     #                     title="Per Partition Labels Distribution",
#     #                 )
    # fig, ax, df = plot_label_distributions(
    #             fds.partitioners["train"],
    #             label_name="label",
    #             plot_type="heatmap",
    #             size_unit="absolute",
    #             partition_id_axis="x",
    #             legend=True,
    #             verbose_labels=True,
    #             title="Per Partition Labels Distribution",
    #             plot_kwargs={"annot": True},
    #         )
    # # Save the plot
    # fig.savefig("plots/label_distribution.png")
#     if args.ufedproto:
#         client_train = partition_train_test["train"]
#         client_test = partition_train_test["test"]
#     else:
#         client_test = fds.load_split("test")
    
#     # pytorch_transforms = transforms.Compose(
#     #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     # )
#     pytorch_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#     def apply_transforms(batch):
#        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
#        return batch
#     # Apply the transforms
#     client_train = client_train.with_transform(apply_transforms)
#     client_test = client_test.with_transform(apply_transforms)
#     print(client_train["image"].shape)
#     exit()
#     # Prepare the DataLoader
#     trainloader = DataLoader(client_train, batch_size=args.batchsize)
#     testloader = DataLoader(client_test, batch_size=args.batchsize)
  
#     return trainloader, testloader


# def load_cifar10_partition(args, partition_id: int):

#     merger = Merger(
#             merge_config={
#                 "train": ("train", "test"),
#                 })
    
#     fds = FederatedDataset(
#         dataset="cifar10",
#         preprocessor= merger if args.ufedproto else None,
#         partitioners={
#             "train": IidPartitioner(
#                         num_partitions=args.clients,
#                         #partition_by="label",
#                     ) if args.isiid else DirichletPartitioner(
#                         num_partitions=args.clients,
#                         partition_by="label",
#                         alpha=args.alpha,
#                         seed=args.seed,
#                         min_partition_size=500,
#                         self_balancing=True
#                     ),
#         },
#     )

#     client_train = fds.load_partition(partition_id, split="train")
    
#     partition_train_test = client_train.train_test_split(test_size=args.split, seed=args.seed)
    
#     fig, ax, df = plot_label_distributions(
#             fds.partitioners["train"],
#             label_name="label",
#             plot_type="heatmap",
#             size_unit="absolute",
#             partition_id_axis="x",
#             legend=True,
#             verbose_labels=True,
#             title="Per Partition Labels Distribution",
#             plot_kwargs={"annot": True},
#         )

#     if args.ufedproto:
#         client_train = partition_train_test["train"]
#         client_test = partition_train_test["test"]
#     else:
#         client_test = fds.load_split("test")


#     pytorch_transforms = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     # Apply the transforms
#     client_train = client_train.with_transform(apply_transforms)
#     client_test = client_test.with_transform(apply_transforms)

#     # Prepare the DataLoader
#     trainloader = DataLoader(client_train, batch_size=args.batchsize)
#     testloader = DataLoader(client_test, batch_size=args.batchsize)
  
#     return trainloader, testloader

def are_models_equal(model1, model2):
    # Compare the state_dict of both models
    for param1, param2 in zip(model1.state_dict().values(), model2.state_dict().values()):
        if not torch.equal(param1, param2):
            return False
    return True


def convert_to_hf_dataset(train_data, test_data=None):
    
    data_dict = {
        "image": [],
        "label": []
    }
    
    for image, label in train_data:
        data_dict["image"].append(image)  # Convert to tensor
        data_dict["label"].append(label)
    
    if test_data is not None:
        for image, label in test_data:
            data_dict["image"].append(image)
            data_dict["label"].append(label)

    return data_dict

def save_label_distribution(args, partitioner):
    
    fig, ax, df = plot_label_distributions(
        partitioner,
        
        label_name="label",
        plot_type="heatmap",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=False,
        title="Per Partition Labels Distribution",
        plot_kwargs={"annot": True},
    )
    # Save the plot
    fig.savefig("plots/"+args.data+"_label_distribution.png")
    plt.close(fig)