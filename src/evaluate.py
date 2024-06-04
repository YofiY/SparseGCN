import os
import pandas as pd
import torch
import time
from src import train
from torch_geometric.logging import init_wandb, log

@torch.no_grad()
def test(model, data, train_mask, test_mask):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [train_mask, test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def eval_model(model, data, train_mask, test_mask, optimizer, N_EPOCHS):
    best_val_acc = 0
    test_acc = 0
    accs = []
    times = []
    start_overall_time = time.time() 
    for epoch in range(1, N_EPOCHS + 1):
        epoch_train_start = time.time()
        loss = train(model, optimizer, data, train_mask)
        train_acc, test_acc = test(model, data, train_mask, test_mask)
        best_val_acc = max(train_acc, best_val_acc)
        accs.append(test_acc)
        times.append(time.time() - epoch_train_start)
        
        if (epoch%10 == 0 or epoch == 1):
            log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    median_time_per_epoch = torch.tensor(times).median()  
    overall_time = time.time() - start_overall_time
    
    return best_val_acc, accs, median_time_per_epoch, overall_time

def snapshot(model_name, k_idx, training_time, training_time_per_epoch, accuracies, best_accuracies, num_edges, sparsification_time):  
    snapshot_path = f'stats/{model_name}/snapshot-at-k_idx{k_idx}' 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    #training_time
    df = pd.DataFrame(training_time)
    df.to_csv(f'{snapshot_path}/training_time.csv')
    
    #training_time_per_epoch
    df = pd.DataFrame(training_time_per_epoch) 
    df.to_csv(f'{snapshot_path}/training_time_per_epoch.csv')
    
    #accuracies
    df = pd.DataFrame(accuracies) 
    df.to_csv(f'{snapshot_path}/accuracies.csv')
    
    #best_accuracies
    df = pd.DataFrame(best_accuracies) 
    df.to_csv(f'{snapshot_path}/best_accuracies.csv')
    
    #num_edges 
    df = pd.DataFrame(num_edges) 
    df.to_csv(f'{snapshot_path}/num_edges.csv')
    
    #sparsification_time
    if (model_name == 'full_graph'): 
        df = pd.DataFrame(sparsification_time) 
        df.to_csv(f'{snapshot_path}/knn_weight_matrix_construction_time.csv')
    else:
        df = pd.DataFrame(sparsification_time) 
        df.to_csv(f'{snapshot_path}/sparsification_time.csv')
    