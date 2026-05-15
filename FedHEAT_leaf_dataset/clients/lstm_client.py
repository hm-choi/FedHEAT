#!/usr/bin/env python
# coding: utf-8
import copy
import time
import gc
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter

from clients.build import CLIENT_REGISTRY

logger = logging.getLogger(__name__)


def shakespeare_collate_fn(batch, pad_idx=0):
    xs, ys = zip(*batch)

    xs = [
        x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)
        for x in xs
    ]
    ys = [
        y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        for y in ys
    ]

    xs = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    ys = torch.stack([y.long() if y.dim() == 0 else y.view(-1).long()[0] for y in ys])

    return xs, ys


@CLIENT_REGISTRY.register()
class LSTM_Client:

    def __init__(self, args, client_index, model=None, loader=None):
        self.args = args
        self.client_index = client_index
        self.model = model

        self.global_model = copy.deepcopy(model)
        for par in self.global_model.parameters():
            par.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
    
    def setup(self, state_dict, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
        self._update_model(state_dict)
        self._update_global_model(state_dict)
        self.device = device
        self.trainer = trainer
        self.num_layers = self.model.num_layers

        train_sampler = None

        pad_idx = getattr(self.args.dataset, "pad_idx", 0)

        self.loader = DataLoader(
            local_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=lambda batch: shakespeare_collate_fn(batch, pad_idx=pad_idx),
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=local_lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.wd,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch,
        )

    def _update_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def __repr__(self):
        n_data = len(self.loader.dataset) if self.loader is not None else 0
        return f"{self.__class__.__name__}(client_index={self.client_index}, data={n_data})"

    def get_weights(self, epoch=None):
        weights = {
            "cls": self.args.client.ce_loss.weight,
        }

        if self.args.client.get("prox_loss"):
            weights["prox"] = self.args.client.prox_loss.weight

        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        self.model.to(self.device)
        self.global_model.to(self.device)

        scaler = GradScaler()
        start = time.time()

        loss_meter = AverageMeter("Loss", ":.2f")
        time_meter = AverageMeter("BatchTime", ":3.1f")

        self.weights = self.get_weights(epoch=global_epoch)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            for i, (tokens, labels) in enumerate(self.loader):
                tokens = tokens.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses = self._algorithm(tokens, labels)
                    loss = sum(self.weights[k] * losses[k] for k in losses)

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), tokens.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()

        logger.info(
            f"[C{self.client_index}] End. Time: {time.time() - start:.2f}s, Loss: {loss_meter.avg:.3f}"
        )

        self.model.to("cpu")
        self.global_model.to("cpu")

        loss_dict = {
            f"loss/{self.args.dataset.name}": loss_meter.avg,
        }

        gc.collect()

        return self.model.state_dict(), loss_dict

    def _algorithm(self, tokens, labels) -> Dict:
        losses = defaultdict(float)

        results = self.model(tokens)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        prox_loss = 0.0

        # FedProx
        if self.args.client.get("prox_loss"):
            fixed_params = {n: p for n, p in self.global_model.named_parameters()}
            for n, p in self.model.named_parameters():
                prox_loss += ((p - fixed_params[n].detach()) ** 2).sum()
            losses["prox"] = prox_loss

        del results
        return losses