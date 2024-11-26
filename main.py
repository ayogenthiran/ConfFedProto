import argparse
import random
import pickle
import torch
import os
from utils.data_utils import load_dataset
from utils.train_utils import train, test, model_fedavg
from utils.model import MnistNet, Cifar10Net
import numpy as np

def main():
    args = parse_arguments()
    print_experiment_info(args)

    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('protos', exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    models = {'mnist': MnistNet, 'cifar10': Cifar10Net}
    global_model = models[args.data]().to(args.device)
    clients = [global_model for _ in range(args.clients)]
    global_proto = {}

    # Initialize metrics storage
    metrics = {
        'acc': [],
        'loss': [],
        'train_acc': [],
        'train_loss': []
    }

    # Store client distributions (only need to do this once)
    if args.plot_label_dist:
        client_distributions = []
        for client_idx in range(args.clients):
            train_loader, _ = load_dataset(args, client_idx)
            dist = get_client_distribution(train_loader)
            client_distributions.append(dist)
        
        with open(f'results/{args.data}_distribution.pkl', 'wb') as f:
            pickle.dump(client_distributions, f)

    for _round in range(args.round):
        round_clients = random.sample(range(len(clients)), int(args.clients * args.clsplit))
        client_protos = {}
        client_metrics = {'test_acc': 0, 'test_loss': 0, 'train_acc': 0, 'train_loss': 0}

        for _client in round_clients:
            train_loader, test_loader = load_dataset(args, _client)
            
            if args.fedproto:
                client_model, protos, tr_acc, tr_loss = train(
                    args, clients[_client], train_loader, _round, _client, global_proto
                )
                ts_loss, ts_acc, test_protos, test_labels = test(args, client_model, test_loader)
                
                update_client_metrics(client_metrics, ts_acc, ts_loss, tr_acc, tr_loss)
                clients[_client].load_state_dict(client_model.state_dict())
                update_client_protos(client_protos, protos)
            else:
                client_model, tr_acc, tr_loss = train(
                    args, clients[_client], train_loader, _round, global_proto
                )
                ts_loss, ts_acc, _, _ = test(args, client_model, test_loader)
                clients[_client] = client_model

        global_model = model_fedavg(clients, global_model, round_clients)
        
        if args.fedproto:
            update_global_proto(global_proto, client_protos)
            round_metrics = {
                'acc': client_metrics['test_acc']/len(round_clients),
                'loss': client_metrics['test_loss']/len(round_clients),
                'train_acc': client_metrics['train_acc']/len(round_clients),
                'train_loss': client_metrics['train_loss']/len(round_clients)
            }
            print_round_metrics(args, _round, client_metrics, len(round_clients))
        else:
            round_metrics = {
                'acc': ts_acc,
                'loss': ts_loss,
                'train_acc': tr_acc,
                'train_loss': tr_loss
            }
            print(f"[FL] Global round {_round+1}: loss: {ts_loss:.4f}, accuracy: {ts_acc:.4f}")

        # Update metrics
        for key in metrics:
            metrics[key].append(round_metrics[key])

    # Save final metrics
    save_metrics(args, metrics)
    print(f"Training completed. Metrics saved for {'FedProto' if args.fedproto else 'Standard FL'}")

def get_client_distribution(dataloader):
    """Calculate class distribution for a client's data"""
    distribution = torch.zeros(10)
    for batch in dataloader:
        labels = batch['label']
        for label in labels:
            distribution[label] += 1
    return distribution.numpy()

def save_metrics(args, metrics):
    """Save training metrics"""
    method = "fedproto" if args.fedproto else "fl"
    filename = f'results/{args.data}_{method}_metrics.pkl'
    print(f"Saving {method} metrics to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning with Prototypes')
    # Regular arguments
    parser.add_argument('--clients', type=int, default=10, help='number of clients')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.07, help='Dirichlet alpha parameter')
    parser.add_argument('--split', type=float, default=0.2, help='train/test split ratio')
    parser.add_argument('--epoch', type=int, default=1, help='local epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--round', type=int, default=20, help='number of rounds')
    parser.add_argument('--clsplit', type=float, default=1.0, help='fraction of clients per round')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--k', type=int, default=10, help='number of classes')
    parser.add_argument('--ld', type=float, default=0.5, help='lambda for prototype loss')
    parser.add_argument('--momentum', type=float, default=0.95, help='SGD momentum')
    
    # Boolean flags with store_true action
    parser.add_argument('--isiid', action='store_true', default=False, help='use IID data distribution')
    parser.add_argument('--fedproto', action='store_true', default=False, help='use federated prototyping')
    parser.add_argument('--pfl', action='store_true', default=False, help='use personalized FL')
    parser.add_argument('--plot_label_dist', action='store_true', default=False, help='plot label distribution')
    parser.add_argument('--clog', action='store_true', default=False, help='enable client logging')
    parser.add_argument('--log', action='store_true', default=False, help='enable general logging')
    
    return parser.parse_args()

def update_client_metrics(metrics, ts_acc, ts_loss, tr_acc, tr_loss):
    metrics['test_acc'] += ts_acc
    metrics['test_loss'] += ts_loss
    metrics['train_acc'] += tr_acc
    metrics['train_loss'] += tr_loss

def update_client_protos(client_protos, protos):
    for key, value in protos.items():
        if key in client_protos:
            client_protos[key].append(value)
        else:
            client_protos[key] = [value]

def update_global_proto(global_proto, client_protos):
    for key in client_protos:
        global_proto[key] = torch.stack(client_protos[key]).mean(dim=0)

def save_client_data(args, client, round, protos, labels):
    with open(f"protos/{args.data}_client_ufedp_{client}_proto_round_{round}.pkl", "wb") as f:
        pickle.dump(protos, f)
    with open(f"protos/{args.data}_client_ufedp_{client}_labels_round_{round}.pkl", "wb") as f:
        pickle.dump(labels, f)

def save_global_proto(args, round, global_proto):
    with open(f"protos/{args.data}_global_proto_{round}.pkl", "wb") as f:
        pickle.dump(global_proto, f)

def print_experiment_info(args):
    print("-" * 23)
    print(f"Dataset: {args.data}")
    print(f"iid: {args.isiid}")
    print(f"Fedproto: {args.fedproto}")
    print(f"proto loss ld: {args.ld}")
    print(f"pfl: {args.pfl}")
    print(f"k: {args.k}")
    print("-" * 23)
    print(f"Total clients: {args.clients}")
    print(f"Global rounds: {args.round}")
    print(f"Local epochs: {args.epoch}")
    print(f"Batch size: {args.batchsize}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Alpha: {args.alpha}")
    print(f"Split: {args.split}")
    print(f"Client split: {args.clsplit}")

def print_round_metrics(args, round, metrics, num_clients):
    if args.fedproto:
        print(f"[PFL] Global round: {round+1}, "
              f"Train loss: {metrics['train_loss']/num_clients:.4f}, "
              f"Test loss: {metrics['test_loss']/num_clients:.4f}, "
              f"Train accuracy: {metrics['train_acc']/num_clients:.4f}, "
              f"Test accuracy: {metrics['test_acc']/num_clients:.4f}")
    else:
        print(f"[FL] Global round: {round+1}, "
              f"Train loss: {metrics['train_loss']/num_clients:.4f}, "
              f"Test loss: {metrics['test_loss']/num_clients:.4f}, "
              f"Train accuracy: {metrics['train_acc']/num_clients:.4f}, "
              f"Test accuracy: {metrics['test_acc']/num_clients:.4f}")

if __name__ == "__main__":
    main()