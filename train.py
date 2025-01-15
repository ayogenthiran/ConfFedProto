import os
import copy
import argparse
import logging
import random
import pickle

#torch
import torch
from torchvision import transforms

#utils
from utils.data_utils import *
from utils.train_utils import *
from utils.model import *
from utils.args_util import *


def main(args):
    # collect args
    #args = parse_arguments()
    print_argsinfo(args)
    # global model
    path = f"models/{args.data}_init_global_model.pth"
    match args.data:
        case "mnist":
            global_model = MnistNet().to(args.device)
        case "cifar10":
            global_model = Cifar10Net().to(args.device)
        case "emnist":
            global_model = EMNISTNet().to(args.device)
        case _:
            raise ValueError(f"Unknown data: {args.data}")

    if os.path.exists(path):
        global_model.load_state_dict(torch.load(path, weights_only=True))
    else:
        torch.save(global_model.state_dict(), path)
    
    total_params = sum(p.numel() for p in global_model.parameters())
    
    # initalize client models with global model
    clients = [copy.deepcopy(global_model).to(args.device) for _ in range(args.clients)]
    
    # initialize global  prototype
    torch.manual_seed(args.seed )
    global_proto = {}  # Shape: [k, 120] 

    # loader train_loader, testloader = load_dataset(args, _client)
    train_loader = [] 
    testloader = []
    for _client in range(args.clients):
        #print(f"Loading data of client: {_client} ")
        _train_loader, _test_loader = load_dataset(args, _client)
        train_loader.append(_train_loader)
        testloader.append(_test_loader)
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
        
            # train the client model
            if args.fedproto:
                # train the client model with fedproto
                _client_model_trained, protos, _accc, _losss = train(args, clients[_client], train_loader[_client], _round, _client, global_proto)
                clients[_client] = _client_model_trained
                
                # collect train and test loss and accuracy
                _ts_loss, _ts_acc, test_protos, test_labels = test(args, _client_model_trained, testloader[_client])
                _tr_loss, _tr_acc, _, _ = test(args, _client_model_trained, train_loader[_client])
                
                if args.clog:
                    # save client test protos and labels
                    with open(f"/scratch/joet/fedproto/protos/{args.data}_client_ufedp_{_client}_proto_round_{_round}.pkl", "wb") as f:
                        pickle.dump(test_protos, f)

                    with open(f"/scratch/joet/fedproto/protos/{args.data}_client_ufedp_{_client}_labels_round_{_round}.pkl", "wb") as f:
                        pickle.dump(test_labels, f)

                client_test_acc += _ts_acc
                client_test_loss += _ts_loss

                client_train_acc += _tr_acc
                client_train_loss += _tr_loss
                
                # collect client prototypes
                for key in protos.keys():
                    if key in client_protos.keys():
                        client_protos[key].append(protos[key])
                    else:
                        client_protos[key] = [protos[key]]
            else:
                _client_model_trained, tr_acc, tr_loss = train(args, _client_model, train_loader[_client],_round, global_proto)
                clients[_client] = _client_model_trained
                # Running average of the models
                # if not args.fedproto:
                #     global_model = running_model_avg(global_model, clients, round_clients)
        
        
        # average the client prototypes and update the global prototype
        if args.fedproto:
            for key in client_protos.keys():
                global_proto[key] = torch.stack(client_protos[key]).mean(dim=0)
        else:
            # compute the global model
            global_dict = server_aggregate_avg(global_model, clients, round_clients)
            global_model.load_state_dict(global_dict)
        
        if args.clog:
            # save prototype to a file
            with open(f"protos/global_proto_{_round}.pkl", "wb") as f:
                pickle.dump(global_proto, f)

        if args.fedproto or args.pfl:
            print(f"[PFL] Global round: {_round+1}, Train loss: {client_train_loss/len(round_clients):.4f}, Test loss: {client_test_loss/len(round_clients):.4f}, Train accuracy: {client_train_acc/len(round_clients):.4f}, Test accuracy: {client_test_acc/len(round_clients):.4f}")
        else:
            #global_model.load_state_dict(running_avg)
            _loss, _acc, _, _ = test(args, global_model, testloader[_client])

            print(f"[RFL]Global round R-FL {_round+1} loss: {_loss}, accuracy: {_acc}")

        # # save the global prototype
        # with open(f"/scratch/joet/fedproto/protos/{args.data}_global_proto_{_round}.pkl", "wb") as f:
        #     pickle.dump(global_proto, f)



if __name__ == "__main__":
    args = parse_arguments()  # default args.
    #print_argsinfo(args)
    #print(configurations)
    main(args)
    # for config in configurations:
    #     args = argparse.Namespace(**config)
    #     main(args)
