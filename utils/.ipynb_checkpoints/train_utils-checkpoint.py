
import torch

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch.nn.functional as F
import time


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net

def test(args, net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    # Initialize lists to collect prototypes and labels
    protos_list = []
    labels_list = []

    with torch.no_grad():
        for batch in testloader:
            match args.data:
                case "mnist":
                    images, labels = batch["image"], batch["label"]
                case "cifar10":
                    images, labels = batch["img"], batch["label"]
                case _:
                    raise ValueError(f"Unknown dataset: {args.dataset}")
            
            images, labels = images.to(args.device), labels.to(args.device)
            outputs, protos = net(images)
            
            # Collect prototypes and labels
            protos_list.append(protos)
            labels_list.append(labels)


            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Concatenate all prototypes and labels into tensors
    test_protos = torch.cat(protos_list, dim=0)
    test_labels = torch.cat(labels_list, dim=0)

    loss /= len(testloader.dataset)
    accuracy = correct / total
    if args.clog:
        print(f"Test loss {loss}, accuracy {accuracy}")
    return loss, accuracy, test_protos, test_labels

def train(args, net, trainloader, global_proto=None):
    
    """
    Train the network on the training set.
    
    """
    
    net.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.epoch):
        
        correct, total, epoch_loss = 0, 0, 0.0

        # initialize client prototypes
        client_protos = {}

        for batch in trainloader:
            match args.data:
                case "mnist":
                    images, labels = batch["image"], batch["label"]
                case "cifar10":
                    images, labels = batch["img"], batch["label"]
                case _:
                    raise ValueError(f"Unknown dataset: {args.dataset}")
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs, proto = net(images)

            # loss is the sum of the classification loss and the prototype loss
            loss1 = criterion(outputs, labels)
            
            # Initialize loss2 to zero
            loss2 = 0 * loss1
            
            # Compute loss2 only if both pfl and ufedproto flags are enabled
            if args.ufedproto:
                # Stack the global prototypes into a tensor
                global_proto_tensor = torch.stack(
                    [global_proto[i] for i in global_proto.keys()],
                    dim=0
                )  # Shape: [k, 120]
                # Compute the prototype loss
                if not args.pfl:
                    # Find the closest prototype labels
                    proto_labels = closest_prototype_labels(proto, global_proto_tensor)
                    loss2 = computeProtoLoss(args, proto, global_proto, proto_labels, loss1)
            
            # Compute the total loss
            loss = loss1 + (args.ld * loss2)

            
            loss.backward()
            optimizer.step()

            # collect client prototypes on the last epoch only to save computation time.
            if epoch == args.epoch - 1:
                if args.ufedproto and not args.pfl:
                    for i, plabel in enumerate(proto_labels):
                        if plabel.item() in client_protos.keys():
                            client_protos[plabel.item()].append(proto[i].detach().cpu())
                        else:
                            client_protos[plabel.item()] = [proto[i].detach().cpu()]

            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

           
        if args.clog:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}, loss1 {loss1}, loss2 {loss2}")
    
    # average the client prototypes
    for key in client_protos.keys():
        client_protos[key] = torch.stack(client_protos[key]).mean(dim=0)
    
    if args.ufedproto:
        return net, client_protos
    else:
        return net
def closest_prototype_labels(input_vectors, prototypes):
    
    """
    Assigns a label to each input vector based on the closest prototype.

    Args:
        input_vectors (torch.Tensor): Tensor of shape [batch_size, 120]
        prototypes (torch.Tensor): Tensor of shape [k, 120]

    Returns:
        torch.Tensor: Tensor of shape [batch_size] containing the assigned labels
    """
    # Ensure prototypes are on the same device and dtype as input_vectors
    prototypes = prototypes.to(device=input_vectors.device, dtype=input_vectors.dtype)

    # Compute pairwise Euclidean distances using torch.cdist
    # input_vectors: [batch_size, 120]
    # prototypes: [k, 120]
    # distances: [batch_size, k]
    #print(input_vectors.shape, len(prototypes))
    #print(prototypes)
    distances = torch.cdist(input_vectors, prototypes, p=2)  # Euclidean distance
    #print(distances)
    # Assign labels based on the closest prototype
    labels = torch.argmin(distances, dim=1)  # [batch_size]
    print(torch.argmin(distances, dim=1))
    
    return labels 

def computeProtoLoss(args, proto, global_proto, labels, loss1):
    
    """
    - Compute the prototype loss between the client prototypes 
      and the global prototype.
    """
    
    proto_loss = 0
    loss_mse = torch.nn.MSELoss().to(args.device)

    if len(global_proto.keys()) == 0:
        loss2 = 0*loss1
        return loss2
    else:
        batch_global_proto = []
        for label in labels:
            if label.item() in global_proto.keys():
                # get the global prototype for the current label.
                label_proto = global_proto[label.item()] 
                batch_global_proto.append(label_proto)
            else:
                # if the prototype for the current label is not available, use zero vector.
                batch_global_proto.append(torch.zeros(len(proto[0])))
    
        batch_global_proto = torch.stack(batch_global_proto)
        
        # compute loss 2
        proto_loss = loss_mse(proto, batch_global_proto.to(args.device))
    
    return proto_loss
    

def fed_average(models):
    """Compute the average of the given models."""
    # Use the state_dict() method to get the parameters of a model
    avg_state = sum([model.state_dict() for model in models]) / len(models)
    return avg_state



def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current