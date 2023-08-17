import glob
import os
import logging
import argparse
from datetime import datetime

from cosense3d.model import get_model
from cosense3d.dataset import get_dataloader
from cosense3d.utils.train_utils import *
from cosense3d.utils.logger import LogMeter
from cosense3d.utils.misc import ensure_dir, setup_logger
from cosense3d.config import load_config, save_config


def train(cfgs):
    seed_everything(2022)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = get_dataloader(cfgs['DATASET'])
    model = get_model(cfgs['MODEL'], 'train').to(device)
    optimizer, lr_scheduler = build_optimizer(model, cfgs['TRAIN']['optimizer'])

    # resume and log_dir
    if cfgs['TRAIN']['resume']:
        log_path = cfgs['TRAIN']['log_dir']
        ckpts = glob.glob(os.path.join(log_path, 'epoch*.pth'))
        epochs = [int(os.path.basename(ckpt)[5:-4]) for ckpt in ckpts]
        max_idx = epochs.index(max(epochs))
        ckpt = ckpts[max_idx]
        logging.info(f"Resuming the model from checkpoint: {ckpt}")
        ckpt = torch.load(ckpt)
        load_model_dict(model, ckpt['model_state_dict'])
        epoch_start = ckpt['epoch']
        lr_scheduler.last_epoch = epoch_start
        # iteration = ckpt['iteration']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        now = datetime.now().strftime('%m-%d-%H-%M-%S')
        run_name = cfgs['TRAIN']['run_name'] + '_' + now
        log_path = os.path.join(cfgs['TRAIN']['log_dir'], run_name)
        ensure_dir(log_path)
        epoch_start = 0

    for m in model.modules:
        module = getattr(model, m)
        if not isinstance(module, torch.nn.Module):
            continue
        mem_params = sum([param.nelement() * param.element_size() for param in module.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
        mem = mem_params + mem_bufs
        logging.info(f'Model.{m} size: {mem / (1024**2):.2f}M')

    logging.info(f'logging to path : {log_path}')

    # get logger
    total_iterations = len(train_dataloader.dataset) // cfgs['DATASET']['batch_size_train']
    logger = LogMeter(total_iterations, log_path, log_every=cfgs['TRAIN']['log_every'],
                      wandb_project=cfgs['TRAIN'].get('project_name', None))

    if not cfgs['TRAIN']['resume']:
        save_config(cfgs, log_path)

    with torch.autograd.set_detect_anomaly(True):
        logging.info('Start training.')
        for epoch in range(epoch_start, cfgs['TRAIN']['max_epoch']):
            train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, logger=logger)
            lr_scheduler.step()
            # train_dataloader.dataset.shuffle_samples()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, logger=None):
    iteration = 1
    len_data = len(train_dataloader)
    model.train()

    for batch_data in train_dataloader:
        load_tensors_to_gpu(batch_data)
        batch_data['epoch'] = epoch
        optimizer.zero_grad()

        # Forward pass
        batch_data = model(batch_data)
        # Loss
        loss, loss_dict = model.loss(batch_data)
        # Gradients
        loss.backward()
        # Updating parameters
        optimizer.step()

        torch.cuda.empty_cache()
        iteration += 1

        # Log training
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) \
            else v for k, v in loss_dict.items() if v}
        if logger is not None:
            logger.log(epoch, iteration, lr_scheduler.get_last_lr()[0], **loss_dict)

        if (iteration + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict['total'],
            }, os.path.join(logger.log_path, f'last.pth'))

    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_dict['total'],
    }, os.path.join(logger.log_path, f'epoch{epoch}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.resume:
        assert os.path.exists(os.path.join(args.log_dir, 'config.yaml')), \
            "config.yaml file not found in the given log_dir."
        setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_config(args)
    setup_logger(args.run_name, args.debug)
    # for ME
    os.environ['OMP_NUM_THREADS'] = "16"
    train(cfgs)