import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def train(args, net, trainloader, global_round, _client, global_proto=None):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr * (0.99 ** global_round),
        momentum=args.momentum,
        weight_decay=1e-4
    )
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=args.epoch)
    scaler = GradScaler() if args.device == 'cuda' else None
    client_protos = {}
    
    for epoch in range(args.epoch):
        correct, total, epoch_loss = 0, 0, 0.0
        
        for batch_idx, batch in enumerate(trainloader):
            try:
                images, labels = batch["image"].to(args.device), batch["label"].to(args.device)
                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with autocast():
                        outputs, proto = net(images)
                        loss1 = criterion(outputs, labels)
                        loss2 = computeProtoLoss(args, proto, global_proto, labels, loss1) if args.fedproto else 0
                        alpha = global_round/args.round
                        loss = loss1 + (alpha * args.ld * loss2)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs, proto = net(images)
                    loss1 = criterion(outputs, labels)
                    loss2 = computeProtoLoss(args, proto, global_proto, labels, loss1) if args.fedproto else 0
                    alpha = global_round/args.round
                    loss = loss1 + (alpha * args.ld * loss2)
                    loss.backward()
                    optimizer.step()

                if epoch == args.epoch - 1 and args.fedproto:
                    for i, label in enumerate(labels):
                        label_item = label.item()
                        if label_item in client_protos:
                            client_protos[label_item].append(proto[i].detach().cpu())
                        else:
                            client_protos[label_item] = [proto[i].detach().cpu()]

                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

                if batch_idx % 50 == 0 and args.clog:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        scheduler.step()
        epoch_acc = correct / total
        epoch_loss = epoch_loss / len(trainloader.dataset)

        if args.clog:
            print(f"Epoch {epoch+1}: loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Average client prototypes
    if args.fedproto:
        for key in client_protos:
            client_protos[key] = torch.stack(client_protos[key]).mean(dim=0)
        return net, client_protos, epoch_acc, epoch_loss
    
    return net, epoch_acc, epoch_loss

def test(args, net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    protos_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in testloader:
            try:
                images = batch["image"].to(args.device)
                labels = batch["label"].to(args.device)
                
                outputs, protos = net(images)
                protos_list.append(protos)
                labels_list.append(labels)
                
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"Error during testing: {str(e)}")
                continue
    
    if total == 0:
        return float('inf'), 0, None, None
    
    test_protos = torch.cat(protos_list, dim=0) if protos_list else None
    test_labels = torch.cat(labels_list, dim=0) if labels_list else None
    
    return loss/total, correct/total, test_protos, test_labels

def computeProtoLoss(args, proto, global_proto, labels, loss1):
    if not global_proto:
        return 0
    
    loss_mse = torch.nn.MSELoss().to(args.device)
    batch_global_proto = []
    
    try:
        for label in labels:
            label_item = label.item()
            if label_item in global_proto:
                batch_global_proto.append(global_proto[label_item])
            else:
                batch_global_proto.append(torch.zeros_like(proto[0]))
        
        batch_global_proto = torch.stack(batch_global_proto).to(args.device)
        return loss_mse(proto, batch_global_proto)
    
    except Exception as e:
        print(f"Error computing proto loss: {str(e)}")
        return 0

def model_fedavg(client_models, globalmodel, round_clients):
    try:
        global_state = globalmodel.state_dict()
        avg_state = {key: torch.zeros_like(value) for key, value in global_state.items()}
        
        for client_idx in round_clients:
            client_state = client_models[client_idx].state_dict()
            for key in avg_state:
                avg_state[key] += client_state[key]
        
        for key in avg_state:
            avg_state[key] = torch.div(avg_state[key], len(round_clients))
        
        globalmodel.load_state_dict(avg_state)
        return globalmodel
    
    except Exception as e:
        print(f"Error in model averaging: {str(e)}")
        return globalmodel

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net