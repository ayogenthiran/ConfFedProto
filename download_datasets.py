import torchvision

def download_datasets():
    # Download MNIST
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./datasets/mnist/', train=True, download=False)
    torchvision.datasets.MNIST(root='./datasets/mnist/', train=False, download=False)
    
    # Download CIFAR10
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=False)
    torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=False)
    
    print("Downloads completed!")

if __name__ == "__main__":
    download_datasets()