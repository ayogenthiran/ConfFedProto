import argparse
import logging
import random
import copy
import pickle
# from flwr_datasets import FederatedDataset
# from flwr_datasets.visualization import plot_label_distributions

#torch
import torch
from torchvision import transforms

#utils
#from utils.utils_vis import *
from utils.data_utils import *
from utils.train_utils import *
from utils.model import *

def main():

    # collect args
    args = parse_arguments()
    
    # print general info about the experiment
    print("-----------------------")
    print(f"Dataset: {args.data}")
    print(f"iid: {args.isiid}")
    print(f"Fedproto: {args.fedproto}")
    print(f"proto loss ld: {args.ld}")
    print(f"pfl: {args.pfl}")
    print(f"k: {args.k}")
    print("-----------------------")
    print(f"Total number of clients: {args.clients}")
    print(f"Total number of global rounds: {args.round}")
    print(f"Local epochs: {args.epoch}")
    print(f"Batch size: {args.batchsize}")
    print(f"learning rate: {args.lr}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"alpha: {args.alpha}")
    print(f"split: {args.split}")
    print(f"clsplit: {args.clsplit}")


    
    # global model
    match args.data:
        case "mnist":
            global_model = MnistNet().to(args.device)
        case "cifar10":
            global_model = Cifar10Net().to(args.device)
        case _:
            raise ValueError(f"Unknown model: {args.data}")

    # initalize client models with global model
    clients = [global_model for _ in range(args.clients)]
    
    # initialize global  prototype
    torch.manual_seed(args.seed )
    global_proto = {}  # Shape: [k, 120] 
    
    # train FL
    for _round in range(args.round):

        # inialize round client prototypes.
        # Store client prototypes after training.
        client_protos = {}

        # select csplit clients randomly
        random.seed(args.seed)
        round_clients = random.sample(range(len(clients)), int(args.clients*args.clsplit))

        if args.clog:
            print(f"Round {_round} selected clients: {round_clients}")
    
        # collect client test accuracies
        client_test_acc = 0
        client_test_loss = 0

        # collect client train accuracies
        client_train_acc = 0
        client_train_loss = 0

        # train the selected clients
        for _client in round_clients:
            
            if args.clog:
                print(f"Training client {_client}")
            
            if not args.fedproto:
                _client_model = clients[_client]
                _client_model.load_state_dict(global_model.state_dict())
            # check if the global model is correctly set to client model
            if not args.fedproto:
                if not are_models_equal(_client_model, global_model):
                    raise ValueError("Global model is not correctly set to client model")

            train_loader, testloader = load_dataset(args, _client)
        
            # train the client model
            if args.fedproto:
                # train the client model with fedproto
                _client_model_trained, protos, _accc, _losss = train(args, clients[_client], train_loader, _round, _client, global_proto)
                
                # collect train and test loss and accuracy
                _ts_loss, _ts_acc, test_protos, test_labels = test(args, _client_model_trained, testloader)
                _tr_loss, _tr_acc, _, _ = test(args, _client_model_trained, train_loader)
                
                # save client test protos and labels
                with open(f"protos/{args.data}_client_ufedp_{_client}_proto_round_{_round}.pkl", "wb") as f:
                    pickle.dump(test_protos, f)

                with open(f"protos/{args.data}_client_ufedp_{_client}_labels_round_{_round}.pkl", "wb") as f:
                    pickle.dump(test_labels, f)

                client_test_acc += _ts_acc
                client_test_loss += _ts_loss

                client_train_acc += _tr_acc
                client_train_loss += _tr_loss

                # replace client model with trained model
                clients[_client].load_state_dict(_client_model_trained.state_dict())
                
                # collect client prototypes
                for key in protos.keys():
                    if key in client_protos.keys():
                        client_protos[key].append(protos[key])
                    else:
                        client_protos[key] = [protos[key]]
            else:
                _client_model_trained, tr_acc, tr_loss = train(args, _client_model, train_loader,_round, global_proto)
                _loss, _acc, _, _ = test(args, _client_model_trained, testloader)
                print(f"Test loss {_loss}, accuracy {_acc}")
                # evaluate the client model
                clients[_client] = _client_model_trained
                # Running average of the models
                # if not args.fedproto:
                #     global_model = running_model_avg(global_model, clients, round_clients)
        # compute average model
        global_model = model_fedavg(clients, global_model, round_clients)
        
        # average the client prototypes and update the global prototype
        if args.fedproto:
            for key in client_protos.keys():
                global_proto[key] = torch.stack(client_protos[key]).mean(dim=0)
        
        # # save prototype to a file
        # with open(f"protos/global_proto_{_round}.pkl", "wb") as f:
        #     pickle.dump(global_proto, f)

        if args.fedproto or args.pfl:
            print(f"[PFL] Global round: {_round+1}, Train loss: {client_train_loss/len(round_clients):.4f}, Test loss: {client_test_loss/len(round_clients):.4f}, Train accuracy: {client_train_acc/len(round_clients):.4f}, Test accuracy: {client_test_acc/len(round_clients):.4f}")
        else:
            #global_model.load_state_dict(running_avg)
            _loss, _acc, _, _ = test(args, global_model, testloader)

            print(f"[RFL]Global round R-FL {_round+1} loss: {_loss}, accuracy: {_acc}")

        # save the global prototype
        with open(f"protos/{args.data}_global_proto_{_round}.pkl", "wb") as f:
            pickle.dump(global_proto, f)

def parse_arguments():
    
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="A brief description of what the script does.")
    
    # Define command-line arguments
    parser.add_argument('--clients', '--clients', default=10, type=str, help='Total number of clients in FL')
    parser.add_argument('--batchsize', '--batchsize', default=16, type=str, help='Total number of clients in FL')
    parser.add_argument('--isiid', '--isiid', default=False, type=bool, help='Total number of clients in FL')
    parser.add_argument('--seed', '--seed', default=42, type=bool, help='Total number of clients in FL')
    
    parser.add_argument('--alpha', '--alpha', default=0.07, type=int, help='Dritchelet alpha value')
    parser.add_argument('--log', '--log', default=True, type=bool, help='log all outputs')
    parser.add_argument('--clog', '--clog', default=False, type=bool, help='client log')
    parser.add_argument('--split', '--split', default=0.2, type=int, help='train test split ratio of client data' )
    parser.add_argument('--epoch', '--epoch', default=1, type=int, help='total epoch per clients')
    
    parser.add_argument('--lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--device', '--device', default='mps', type=str, help='device to train the model')
    parser.add_argument('--round', '--round', default=20, type=int, help='total number of global rounds')
    parser.add_argument('--clsplit', '--clsplit', default=1, type=float, help='client split for training')
    parser.add_argument('--data', '--data', default='cifar10', type=str, help='Dataset used for training')
    parser.add_argument('--plot_label_dist', '--plot_label_dist', default=True, type=bool, help='Plot label discribution. File will be saved in the plots folder')
    
    #parser.add_argument('-fedproto', '--fedproto', default=True, type=str, help='use federated prototyping')
    parser.add_argument('--k', '--k', default=10, type=float, help='number of prototypes')
    
    parser.add_argument('--fedproto', '--fedproto', default=False, type=bool, help='use federated prototyping')
    parser.add_argument('--pfl', '--pfl', default=False, type=str, help='train pfl without fedproto loss')
    parser.add_argument('--ld', '--ld', default=0.5, type=float, help='lambda value for prototype loss')
    # add momentum argunent
    parser.add_argument('--momentum', '--momentum', default=0.95, type=float, help='momentum value')
   
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
