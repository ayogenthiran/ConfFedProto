import torch

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt


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
            
            images, labels = batch["image"], batch["label"]

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

def train(args, net, trainloader, global_round, _client, global_proto=None):

    """
    Train the network on the training set.
    """
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(args.epoch):
        
        correct, total, epoch_loss = 0, 0, 0.0

        # initialize client prototypes
        client_protos = {}
        
        for batch in trainloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs, proto = net(images)

            # loss is the sum of the classification loss and the prototype loss
            loss1 = criterion(outputs, labels)

            if args.conffedproto:
                # Compute confidence and predicted classes in one step
                probabilities = F.softmax(outputs, dim=1)
                confidence_scores = probabilities[range(labels.size(0)), labels] #confidence_scores = torch.max(probabilities, dim=1).values
                proto = confidence_scores.unsqueeze(1) * proto
                
            # Initialize loss2 to zero
            loss2 = 0 * loss1
            
            # Compute loss2 only if both pfl and ufedproto flags are enabled
            if args.fedproto:
                # Compute the prototype loss
                if not args.pfl:
                    loss2 = computeProtoLoss(args, proto, global_proto, labels, loss1)

            alpha = (global_round/args.round)
            loss = loss1 + ((alpha*args.ld) * loss2)
            
            loss.backward()
            optimizer.step()
        
            # collect client prototypes on the last epoch only to save computation time.
            if epoch == args.epoch - 1:
                if args.fedproto and not args.pfl:
                    for i, plabel in enumerate(labels):
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

        #scheduler.step()

           
        if args.clog:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}, loss1 {loss1}, loss2 {loss2}")
    
    # # plot client prototype label distribution
    # if not args.pfl:
    #     # Plotting the distributions of both lists
    #     plot_proto_label_dist(clients_protos_label, clients_actual_labels, _client, global_round)

    # average the client prototypes
    for key in client_protos.keys():
        client_protos[key] = torch.stack(client_protos[key]).mean(dim=0)
    
    if args.fedproto:
        return net, client_protos, epoch_acc, epoch_loss
    else:
        return net, epoch_acc, epoch_loss


def plot_proto_label_dist(proto_labels, actual_labels, _client, _global_round):
    

    # Step 1: Find unique values in list1 and list2
    unique_protolabel_values = sorted(set(proto_labels))
    unique_actuallabel_values = sorted(set(actual_labels))

    # Step 2: Initialize a data structure to hold counts
    list2_value_to_index = {v: i for i, v in enumerate(unique_actuallabel_values)}
    data = np.zeros((len(unique_actuallabel_values), len(unique_protolabel_values)))

    # Step 3: Populate the data array with counts
    for idx1, val1 in enumerate(unique_protolabel_values):
        indices = [i for i, x in enumerate(proto_labels) if x == val1]
        corresponding_list2 = [actual_labels[i] for i in indices]
        counts = {}
        for val2 in corresponding_list2:
            counts[val2] = counts.get(val2, 0) + 1
        for val2, count in counts.items():
            idx2 = list2_value_to_index[val2]
            data[idx2, idx1] = count

    # Step 4: Plot the stacked bar chart
    labels = [str(val) for val in unique_protolabel_values]
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots()

    bottom = np.zeros(len(unique_protolabel_values))
    colors = plt.cm.tab20.colors  # You can choose a different colormap if you like

    for i in range(len(unique_actuallabel_values)):
        ax.bar(labels, data[i], width, bottom=bottom, label=str(unique_actuallabel_values[i]), color=colors[i])
        bottom += data[i]

    # Step 5: Add labels and legend
    ax.set_xlabel('Values in proto_labels')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of actual proto labels with Distribution of proto labels')
    ax.legend(title='Actual labels Values')
        
    # plt.hist(clients_protos_label, bins=range(min(clients_protos_label), max(clients_protos_label) + 2), align='left', edgecolor='black', alpha=0.5, label='clients_protos_label')
    # # Labeling the plot
    # plt.xlabel('cluster')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of clusters clients_'+str(_client)+'_round_'+str(global_round))
    # plt.legend()
    # plt.xticks(range(min(clients_protos_label), max(clients_protos_label) + 1))
    
    file_path = 'plots/label_dist/client_'+str(_client)+'_round_'+str(_global_round)+'_distribution.png'
    plt.savefig(file_path)
    plt.close()
    fig.close()
    

def closest_prototype_labels(input_vectors, prototypes):
    
    """
    Assigns a label to each input vector based on the closest prototype.

    Args:
        input_vectors (torch.Tensor): Tensor of shape [batch_size, 120]
        prototypes (torch.Tensor): Tensor of shape [k, 120]

    Returns:
        torch.Tensor: Tensor of shape [batch_size] containing the assigned labels
    """
    
    #print(input_vectors.shape, prototypes.shape)

    
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
    #print(torch.argmin(distances, dim=1))
    
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


def server_aggregate_avg(global_model, client_models, selected_clients):
    global_dict = global_model.state_dict()
    
    for k in global_dict.keys():
        # Average the parameters from all selected clients
        global_dict[k] = sum(
            client_models[i].state_dict()[k].float()
            for i in selected_clients
        ) / len(selected_clients)
    
    # Update the global model with the aggregated parameters
    return global_dict