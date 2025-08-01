import torch
from einops import rearrange, repeat

from tqdm import tqdm
import os
import os.path as opt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_one_epoch(epoch, num_epochs, model, mae, optimizer, dataloader, criterion, scheduler, device):
    model.train()
    mae.eval()
    losses = []
    pbar = tqdm(dataloader, total=len(dataloader))
    for d in pbar:
        sequence = d['Sequence'].to(device)  # (B, C, T, V)
        adj_mat  = d['AM'].to(device)        # (B, T, V, V)
        labels   = d['Label'].to(device)

        # --- extract per-node features from MAE encoder ---
        B,C,T,V = sequence.shape
        x = rearrange(sequence, 'b c t v -> (b t) v c')
        a = rearrange(adj_mat,   'b t v v -> (b t) v v')
        feats = mae.encode(x, a)             # (B*T, V, D)
        feats = feats.view(B, T, V, -1)
        tokens = rearrange(feats, 'b t v d -> b d t v').unsqueeze(-1)

        optimizer.zero_grad()
        pred = model(tokens)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg = sum(losses)/len(losses)
        pbar.set_description(f"[{epoch}/{num_epochs}] train loss: {avg:.4f}")

    if scheduler:
        scheduler.step()
    return avg


def valid_one_epoch(model, mae, dataloader, criterion, device):
    model.eval()
    mae.eval()
    losses, correct, total = [], 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for d in pbar:
            sequence = d['Sequence'].to(device)
            adj_mat  = d['AM'].to(device)
            labels   = d['Label'].to(device)

            # same encoding as above
            B,C,T,V = sequence.shape
            x = rearrange(sequence, 'b c t v -> (b t) v c')
            a = rearrange(adj_mat,   'b t v v -> (b t) v v')
            feats = mae.encode(x, a).view(B, T, V, -1)
            tokens = rearrange(feats, 'b t v d -> b d t v').unsqueeze(-1)

            pred = model(tokens)
            loss = criterion(pred, labels)

            losses.append(loss.item())
            avg_loss = sum(losses)/len(losses)

            correct += (pred.argmax(1) == labels).sum().item()
            total   += labels.size(0)
            acc = 100 * correct / total

            all_preds .extend(pred.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_description(f"[VALID] loss: {avg_loss:.4f} acc: {acc:.1f}%")

    return avg_loss, acc, all_labels, all_preds


def eval_stgcn(stgcn, mae, dataloader, device, args):

    save_folder_path = opt.join(args.save_folder_path, args.exp_name)
    os.makedirs(save_folder_path, exist_ok=True)

    accuracy = 0.0
    n = 0
    pred_labels, true_labels = [], []
    stgcn.eval()
    mae.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for d in pbar:
            sequence = d['Sequence'].to(device)
            adj_mat = d['AM'].to(device)
            label = d['Label'].to(device)

            tokens = mae.inference(sequence, adj_mat)
            tokens = rearrange(tokens, 'b t n d -> b d t n')
            tokens = tokens.unsqueeze(-1)
        
            true_labels.extend(label.tolist())
            output = stgcn(tokens)

            accuracy += (output.argmax(dim=1) == label.flatten()).sum().item()
            n += len(label.flatten())
            
            pred_labels.extend(output.argmax(dim=1).tolist())
            desc = '[VALID]> acc. %.2f%%' % ((accuracy / n)*100)
            pbar.set_description(desc)

    accuracy = (accuracy / n) * 100

    ## file results
    results_txt = []
    for idx, label in enumerate(pred_labels, start=1):
        results_txt.append(' '.join([str(idx), dataloader.dataset.label_map[label]]))

    results_txt = '\n'.join(results_txt)

    with open(opt.join(save_folder_path, 'result_file.txt'), 'w') as fd:
        fd.write(results_txt)


    ## confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    norm_cm = confusion_matrix(true_labels, pred_labels, normalize='true')*100

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    sns.heatmap(cm,
                ax=ax1, annot=True, fmt='.3g', 
                xticklabels=dataloader.dataset.label_map,
                yticklabels=dataloader.dataset.label_map)
    ax1.set_title('confusion matrix')

    sns.heatmap(norm_cm,
                ax=ax2, annot=True, fmt='.3g', 
                xticklabels=dataloader.dataset.label_map,
                yticklabels=dataloader.dataset.label_map)
    ax2.set_title('normalized confusion matrix (%)')

    fig.savefig(opt.join(save_folder_path, 'cm_best.png'))
    plt.show()
    
    print('done !')


def training_stgcn_loop(model, mae, train_loader, valid_loader,
                        optimizer, criterion, scheduler, device, args):
    save_folder = os.path.join(args.save_folder_path, args.exp_name, 'weights')
    os.makedirs(save_folder, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.stgcn.num_epochs+1):
        train_loss = train_one_epoch(epoch, args.stgcn.num_epochs,
                                     model, mae, optimizer, train_loader,
                                     criterion, scheduler, device)
        val_loss, acc, _, _ = valid_one_epoch(model, mae, valid_loader,
                                              criterion, device)

        if acc >= best_acc:
            best_acc = acc
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val': best_acc,
            }, os.path.join(save_folder, 'best_stgcn_model.pth'))
            print(f"→ New best STGCN at epoch {epoch}: {acc:.2f}%")

    print("Training complete.")