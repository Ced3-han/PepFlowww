import wandb

import os
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from pepflow.utils.vc import get_version, has_changes
from pepflow.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from pepflow.utils.data import PaddingCollate
from pepflow.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses

from models_con.pep_dataloader import PepDataset
# from models_con.flow_model import FlowModel

from models_con.flow_model import FlowModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/angle/learn_angle.yaml')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--name', type=str, default='pepflow')
    args = parser.parse_args()

    # Version control
    branch, version = get_version()
    version_short = '%s-%s' % (branch, version[:7])
    if has_changes() and not args.debug:
        c = input('Start training anyway? (y/n) ')
        if c != 'y':
            exit()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = args.device

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag))
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, version_short), tag=args.tag)
        with open(os.path.join(log_dir, 'commit.txt'), 'w') as f:
            f.write(branch + '\n')
            f.write(version + '\n')
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    # train_dataset = get_dataset(config.dataset.train)
    # val_dataset = get_dataset(config.dataset.val)
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    # val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
    #                                         name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    # val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(train_dataset)))

    # Model
    logger.info('Building model...')
    # model = get_model(config.model).to(args.device)
    model = FlowModel(config.model).to(args.device)
    # wandb.watch(model,log='all',log_freq=1)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward pass
        # loss_dict, metric_dict = model.get_loss(batch) # get loss and metrics
        loss_dict = model(batch) # get loss and metrics
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        # loss = loss / config.train.accum_grad
        time_forward_end = current_milli_time()

        if torch.isnan(loss):
            print('NAN Loss!')
            torch.save({'batch':batch,'loss':loss,'loss_dict':loss_dict,'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,},os.path.join(log_dir,'nan.pt'))
            loss = torch.tensor(0.,requires_grad=True).to(loss.device)

        loss.backward()

        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        # Backward
        # if it % config.train.accum_grad ==0:
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {}
        # scalar_dict.update(metric_dict['scalar'])
        scalar_dict.update({
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger)

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                # loss_dict, metric_dict = model.get_loss(batch)
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
                for k, v in loss_dict['scalar'].items():
                    scalar_accum.add(name=k, value=v, batchsize=len(batch['aa']), mode='mean')
            
        avg_loss = scalar_accum.get_average('loss')
        summary = scalar_accum.log(it, 'val', logger=logger, writer=writer)
        for k,v in summary.items():
            wandb.log({f'val/{k}': v}, step=it)
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            # if it % config.train.val_freq == 0:
            #     avg_val_loss = validate(it)
                # if not args.debug:
            if it % config.train.val_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    # 'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')