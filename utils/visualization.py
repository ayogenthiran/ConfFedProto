import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

def plot_combined_comparison():
    """Plot accuracy and loss comparisons for both MNIST and CIFAR10"""
    # Create figure with 2 rows (MNIST and CIFAR10)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot for each dataset
    for idx, dataset_name in enumerate(['mnist', 'cifar10']):
        fl_path = f'results/{dataset_name}_fl_metrics.pkl'
        fedproto_path = f'results/{dataset_name}_fedproto_metrics.pkl'
        
        if os.path.exists(fl_path) and os.path.exists(fedproto_path):
            with open(fl_path, 'rb') as f:
                fl_metrics = pickle.load(f)
            with open(fedproto_path, 'rb') as f:
                fedproto_metrics = pickle.load(f)
            
            # Accuracy plot
            axes[idx, 0].plot(fl_metrics['acc'], 'b-o', label='Standard FL')
            axes[idx, 0].plot(fedproto_metrics['acc'], 'r-o', label='FedProto')
            axes[idx, 0].set_xlabel('Communication Round')
            axes[idx, 0].set_ylabel('Test Accuracy')
            axes[idx, 0].set_title(f'{dataset_name.upper()}: Test Accuracy Comparison')
            axes[idx, 0].grid(True)
            axes[idx, 0].legend()
            
            # Loss plot
            axes[idx, 1].plot(fl_metrics['loss'], 'b-o', label='Standard FL')
            axes[idx, 1].plot(fedproto_metrics['loss'], 'r-o', label='FedProto')
            axes[idx, 1].set_xlabel('Communication Round')
            axes[idx, 1].set_ylabel('Test Loss')
            axes[idx, 1].set_title(f'{dataset_name.upper()}: Test Loss Comparison')
            axes[idx, 1].grid(True)
            axes[idx, 1].legend()
            
            print(f"\nFinal Results for {dataset_name.upper()}:")
            print(f"FedProto - Final Accuracy: {fedproto_metrics['acc'][-1]:.4f}")
            print(f"Standard FL - Final Accuracy: {fl_metrics['acc'][-1]:.4f}")
    
    plt.tight_layout()
    plt.savefig('plots/combined_comparison.png')
    plt.close()

def plot_combined_distribution(num_clients=6):
    """Plot non-IID distributions for both MNIST and CIFAR10"""
    fig, axes = plt.subplots(2, num_clients, figsize=(20, 8))
    
    for idx, dataset_name in enumerate(['mnist', 'cifar10']):
        dist_path = f'results/{dataset_name}_distribution.pkl'
        
        if os.path.exists(dist_path):
            with open(dist_path, 'rb') as f:
                client_distributions = pickle.load(f)
            
            for i in range(num_clients):
                dist = client_distributions[i]
                normalized_dist = dist / dist.sum()
                
                axes[idx, i].bar(range(10), normalized_dist)
                axes[idx, i].set_title(f'Client {i+1}')
                axes[idx, i].set_xlabel('Class' if dataset_name == 'cifar10' else 'Digit')
                axes[idx, i].set_ylabel('Proportion')
                axes[idx, i].grid(True, alpha=0.3)
            
            # Add dataset label on the left
            axes[idx, 0].set_ylabel(f'{dataset_name.upper()}\nProportion')
    
    plt.suptitle('Non-IID Distribution Comparison (α=0.3)', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/combined_distribution.png')
    plt.close()

def save_metrics_summary():
    """Save numerical results to a text file"""
    with open('plots/results_summary.txt', 'w') as f:
        for dataset_name in ['mnist', 'cifar10']:
            fl_path = f'results/{dataset_name}_fl_metrics.pkl'
            fedproto_path = f'results/{dataset_name}_fedproto_metrics.pkl'
            
            if os.path.exists(fl_path) and os.path.exists(fedproto_path):
                with open(fl_path, 'rb') as fl_f:
                    fl_metrics = pickle.load(fl_f)
                with open(fedproto_path, 'rb') as fp_f:
                    fedproto_metrics = pickle.load(fp_f)
                
                f.write(f"\nResults for {dataset_name.upper()}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"FedProto - Final Accuracy: {fedproto_metrics['acc'][-1]:.4f}\n")
                f.write(f"Standard FL - Final Accuracy: {fl_metrics['acc'][-1]:.4f}\n")
                f.write(f"Improvement: {(fedproto_metrics['acc'][-1] - fl_metrics['acc'][-1])*100:.2f}%\n")

if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_combined_comparison()
    plot_combined_distribution()
    save_metrics_summary()
    
    print("\nPlots have been saved to the 'plots' directory:")
    print("1. combined_comparison.png - Performance comparison for both datasets")
    print("2. combined_distribution.png - Non-IID distribution visualization")
    print("3. results_summary.txt - Numerical results summary")