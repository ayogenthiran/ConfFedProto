import torchvision

def download_datasets():
    # Download MNIST
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./datasets/mnist/', train=True, download=True)
    torchvision.datasets.MNIST(root='./datasets/mnist/', train=False, download=True)
    
    # Download CIFAR10
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=True)
    torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True)
    
    print("Downloads completed!")

if __name__ == "__main__":
    download_datasets()