import torch

from tqdm import tqdm
import os
import os.path as opt


def train_one_epoch(epoch, num_epochs, model, dataloader, optimizer, scheduler, device):
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    losses = []
    for d in pbar:
        seq = d['Sequence'].to(device)
        A   = d['A_temporal'].to(device)

        optimizer.zero_grad()
        edge_pred, loss = model(seq, A)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg = sum(losses) / len(losses)
        pbar.set_description(f"[{epoch}/{num_epochs}] train loss: {avg:.4f}")

    if scheduler is not None:
        scheduler.step()
    return avg


def valid_one_epoch(model, dataloader, device):
    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    losses = []
    with torch.no_grad():
        for d in pbar:
            seq = d['Sequence'].to(device)
            A   = d['AM'].to(device)
            _, loss = model(seq, A)

            losses.append(loss.item())
            avg = sum(losses) / len(losses)
            pbar.set_description(f"[VALID] loss: {avg:.4f}")
    return avg


def training_mae_loop(model, train_loader, valid_loader, optimizer, scheduler, device, args):

    save_folder_path = opt.join(args.save_folder_path, args.exp_name,'weights/')
    os.makedirs(save_folder_path, exist_ok=True)
    
    ## TRAINING
    start_epoch = 1
    best_val = float("inf")

    ## training loop
    for epoch in range(start_epoch, args.mae.num_epochs + 1):
        train_loss = train_one_epoch(epoch, args.mae.num_epochs, model, train_loader, optimizer, scheduler, device)
        valid_loss = valid_one_epoch(model, valid_loader, device)
        
        is_best = valid_loss < best_val
        best_val = min(valid_loss, best_val)
        
        if is_best:
            torch.save(
                {'state_dict': model.state_dict(),
                 'epoch': epoch,
                 'best_val': best_val
                },
                opt.join(save_folder_path, "best_mae_model.pth"),
            )
            print(f"→ New best at epoch {epoch}, val loss = {best_val:.4f}")