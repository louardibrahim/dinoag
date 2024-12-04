import argparse
import sys
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dinoag.model import DINOAG
from dinoag.data import CARLA_Data
from dinoag.config import GlobalConfig

from pytorch_lightning.strategies import DDPStrategy




import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int):
        super().__init__()
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            prev = (
                Path(self.dirpath) / f"latest-{pl_module.global_step - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            prev.unlink(missing_ok=True)

class DINOAG_planner(pl.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = DINOAG(config)
        self._load_weight()

    def _load_weight(self):
        rl_state_dict = torch.load(self.hparams.config.rl_ckpt, map_location='cpu')['policy_state_dict']
        self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')

    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']

        gt_waypoints = batch['waypoints']

        pred = self.model(front_img, state, target_point)

        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value) ) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature))* self.config.features_weight

        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

        loss =  speed_loss + value_loss + feature_loss + wp_loss 

        self.log('train_wp_loss', wp_loss.item())
        self.log('train_speed_loss', speed_loss.item())
        self.log('train_value_loss', value_loss.item())
        self.log('train_feature_loss', feature_loss.item())

        self.log('loss', loss,on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


    def validation_step(self, batch, batch_idx):
        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']
        gt_waypoints = batch['waypoints']

        pred = self.model(front_img, state, target_point)

        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value)) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature))* self.config.features_weight
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()


        val_loss = wp_loss 


        self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
        self.log('val_value_loss', value_loss.item(), sync_dist=True)
        self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
        self.log('val_wp_loss', wp_loss.item(), sync_dist=True)

        self.log('val_loss', val_loss.item(), sync_dist=True)


#====================

if __name__ == "__main__":
    
    # Define command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='DINOAG', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs.')
    
    args = parser.parse_args()

    # Modify log directory based on experiment ID
    args.logdir = os.path.join(args.logdir, args.id)

    # Initialize global configuration (assumes GlobalConfig class is defined)
    config = GlobalConfig()
    GlobalConfig.initialize_data_urls()

    # Load datasets
    train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=False)
    print(f"Training set size: {len(train_set)}")
    
    val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data)
    print(f"Validation set size: {len(val_set)}")

    # Create data loaders with batch size from command-line arguments
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize the model with learning rate from command-line arguments
    DINOAG_model = DINOAG_planner(config, args.lr)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        mode="max",
        monitor="val_loss",
        save_top_k=2,
        save_last=True,
        dirpath=args.logdir,
        filename="best_{epoch:02d}-{val_loss:.3f}"
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    # Initialize trainer using GPUs and other parameters from command-line arguments
    trainer = pl.Trainer(
        default_root_dir=args.logdir,
        accelerator="gpu",  # Use GPU
        devices=args.gpus,  # Use the number of GPUs specified in args
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_true",  # DataParallel strategy
        profiler='simple',
        callbacks=[PeriodicCheckpoint(1000), checkpoint_callback],
        log_every_n_steps=100,
        check_val_every_n_epoch=args.val_every,  # Validation frequency from args
        max_epochs=args.epochs  # Max epochs from args
    )

    # Start training
    trainer.fit(DINOAG_model, dataloader_train, dataloader_val)


