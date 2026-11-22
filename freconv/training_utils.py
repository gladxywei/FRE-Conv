import os

import torch
import torch.distributed as dist
from torch.cuda import amp

from . import logging, utils
from .progress import progress_bar

__all__ = ["train"]

logger = logging.get_logger()


def _train_loop(train_loader, network, cla, epoch, optimizer, criterion, scaler, iter_num, max_iterations):
    train_loss = 0
    avg_loss = 0
    max_batches = min(len(train_loader), max_iterations - iter_num)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        if not cla.no_cuda and torch.cuda.is_available():
            inputs, targets = inputs.to(cla.device), targets.to(cla.device)
        optimizer.zero_grad()
        loss_data = 0
        with amp.autocast():
            outputs = network(inputs)
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)
            loss = criterion(outputs, targets)


        scaler.scale(loss).backward(retain_graph=True)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1e6)
        loss_data += loss.item()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss_data
        avg_loss = train_loss / (batch_idx + 1)
        progress_bar(batch_idx, max_batches, 'AvgLoss: %.4e | Loss: %.4e' 
                         % (avg_loss, loss_data))
        if iter_num + batch_idx + 1 >= max_iterations:
            break

    iter_num += batch_idx + 1
    logger.info(f"Epoch: {epoch} iter_num: {iter_num} Loss: {avg_loss}")
    return avg_loss, iter_num


def train(train_loader, network, cla, epoch, optimizer, criterion, writer=None, iterations=150):
    logger.info(f"Train:\t\tEpoch: {epoch}")
    network.train()

    scaler = amp.GradScaler()
    it = 0
    while True:
        avg_loss, it = _train_loop(
            train_loader=train_loader,
            network=network,
            cla=cla,
            epoch=epoch,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            iter_num=it,
            max_iterations=iterations,
        )
        if it >= iterations:
            break

    if writer is not None:
        writer.add_scalar(os.path.join(cla.prefix, "train_loss_epoch"), avg_loss, epoch)


def validate(valid_loader, name, network, cla, epoch, criterion, writer=None):
    network.eval()
    validate_loss = 0
    total_psnr = 0
    
    ls, psnrs = [], []
    logger.info(f"Validation:\tEpoch: {epoch} dataset name: {name}")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not cla.no_cuda and torch.cuda.is_available():
                inputs, targets = inputs.to(cla.device), targets.to(cla.device)

            loss_data = 0
            outputs = network(inputs)
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)
            loss = criterion(outputs, targets)
            loss_data += loss.item()

            # data units: [batch, bands, h, w]
            psnr = []
           
            for d in range(outputs.shape[0]):
                # band dim assumes that the batch is > 1
                psnr.append(utils.peak_snr(outputs[d], targets[d], bandwise=False, band_dim=-3))
                
            psnrs.extend(psnr)
            
            psnr = sum(psnr) / float(len(psnr))


            ls.append(loss_data)

            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx + 1)

            total_psnr += psnr
           
            avg_psnr = total_psnr / (batch_idx + 1)


    if dist.is_initialized():
        # average all the results
        sz = dist.get_world_size()
        red = torch.tensor([avg_psnr, avg_loss], device=cla.device) / float(sz)
        dist.all_reduce(red)
        avg_psnr = red[0].item()
        avg_loss = red[1].item()

    logger.info(f"Final: Loss: {avg_loss} | PSNR: {avg_psnr} ")

    if writer is not None:
        writer.add_scalar(os.path.join(cla.prefix, name, "val_loss_epoch"), avg_loss, epoch)
        writer.add_scalar(os.path.join(cla.prefix, name, "val_psnr_epoch"), avg_psnr, epoch)


    avg_psnr = sum(psnrs) / float(len(psnrs))

    avg_loss = sum(ls) / float(len(ls))
    return avg_psnr, avg_loss


def save_checkpoint(cla, epoch, network, optimizer, model_out_path=None, **kwargs):
    if cla.rank != 0:
        return
    if not model_out_path:
        model_out_path = os.path.join(cla.save_dir, cla.arch, f"model_epoch_{epoch}.pth")

    state = {
        "net": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    state.update(kwargs)

    if not os.path.isdir(os.path.join(cla.save_dir, cla.arch)):
        os.makedirs(os.path.join(cla.save_dir, cla.arch))

    torch.save(state, model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
