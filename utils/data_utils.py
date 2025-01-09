import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import matplotlib.pyplot as plt

def load_dataset(args, partition_id: int):
    # Dataset loading
    datasets = {
        'mnist': (torchvision.datasets.MNIST, ((0.1307,), (0.3081,))),
        'cifar10': (torchvision.datasets.CIFAR10, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
    }
    
    if args.data not in datasets:
        raise ValueError(f"Unsupported dataset: {args.data}")
        
    dataset_class, normalize_stats = datasets[args.data]
    trainset = dataset_class(root=f'./datasets/{args.data}/', train=True, download=False)
    testset = dataset_class(root=f'./datasets/{args.data}/', train=False, download=False)

    partitioner = (IidPartitioner if args.isiid else DirichletPartitioner)(
        num_partitions=args.clients,
        partition_by="label",
        alpha=args.alpha if not args.isiid else None,
        seed=args.seed,
        min_partition_size=500,
        self_balancing=True
    )

    # Dataset preparation
    ds_dataset = Dataset.from_dict(convert_to_hf_dataset(trainset, testset if args.fedproto else None))
    partitioner.dataset = ds_dataset

    if args.plot_label_dist:
        save_label_distribution(args, partitioner)

    # Client data splitting
    client_data = partitioner.load_partition(partition_id=partition_id)
    if args.fedproto:
        partition_split = client_data.train_test_split(test_size=args.split, seed=args.seed)
        client_train, client_test = partition_split["train"], partition_split["test"]
    else:
        client_train = client_data
        client_test = Dataset.from_dict(convert_to_hf_dataset(testset))

    # Data transforms
    _transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize_stats)
    ])

    # Apply transforms and create dataloaders
    transform_fn = lambda batch: {"image": [_transforms(img) for img in batch["image"]]}
    client_train = client_train.with_transform(transform_fn)
    client_test = client_test.with_transform(transform_fn)

    return (
        DataLoader(client_train, batch_size=args.batchsize),
        DataLoader(client_test, batch_size=args.batchsize)
    )

def convert_to_hf_dataset(train_data, test_data=None):
    data_dict = {"image": [], "label": []}
    for image, label in train_data:
        data_dict["image"].append(image)
        data_dict["label"].append(label)
    
    if test_data:
        for image, label in test_data:
            data_dict["image"].append(image)
            data_dict["label"].append(label)
    
    return data_dict

def save_label_distribution(args, partitioner):
    fig, ax = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="heatmap",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=False,
        title="Per Partition Labels Distribution",
        plot_kwargs={"annot": True}
    )
    fig.savefig(f"plots/{args.data}_label_distribution.png")
    plt.close(fig)