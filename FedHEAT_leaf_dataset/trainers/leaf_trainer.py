from pathlib import Path
from typing import Dict, Type
from collections import defaultdict
import copy
import gc
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm.auto import trange
import wandb

from torch.utils.data import DataLoader
from omegaconf import DictConfig

from trainers.build import TRAINER_REGISTRY
from servers import Server
from utils import terminate_processes, initalize_random_seed

from servers.he_engine import *

logger = logging.getLogger(__name__)

mp.set_sharing_strategy("file_system")


def local_update(
    device,
    task_queue,
    result_queue,
    args,
    client_type,
    model
):

    if args.multiprocessing:
        torch.cuda.set_device(device)
        initalize_random_seed(args)

    while True:
        task = task_queue.get()
        if task is None:
            break

        client = client_type(
            args,
            client_index=task["client_idx"],
            model=copy.deepcopy(model),
        )

        local_dataset = task["local_dataset"]
        local_dataset.tensorize()

        setup_inputs = {
            "state_dict": task["state_dict"],
            "device": device,
            "local_dataset": local_dataset,
            "local_lr": task["local_lr"],
            "global_epoch": task["global_epoch"],
            "trainer": None,
        }

        client.setup(**setup_inputs)
        
        # Local Training
        local_model, local_loss_dict = client.local_train(global_epoch=task["global_epoch"])
        result_queue.put((local_model, local_loss_dict))

        gc.collect()
        del local_dataset, client, setup_inputs, task
        if not args.multiprocessing:
            break


@TRAINER_REGISTRY.register()
class Leaf_trainer:

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:
        
        self.args = args
        self.device = device
        self.model = model

        self.checkpoint_path = Path(self.args.checkpoint_path)
        mode = self.args.split.mode
        if self.args.split.mode == 'dirichlet':
            mode += str(self.args.split.alpha)
        self.exp_path = self.checkpoint_path / self.args.dataset.name / mode / self.args.exp_name
        logger.info(f"Exp path : {self.exp_path}")

        ### training config
        trainer_args = self.args.trainer
        self.global_rounds = trainer_args.global_rounds
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay

        self.datasets = datasets
        self.train_dataset = self.datasets["train"]
        self.test_dataset = self.datasets["test"]

        self.num_clients = len(self.train_dataset.clients)
        self.client_type = client_type
        self.server = server
        if self.args.server.momentum > 0:
            if self.args.trainer.encrypted == True:
                self.server.encrypted_set_momentum(self.model)
            if self.args.trainer.encrypted == False:
                self.server.set_momentum(self.model)

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        self.eval_params = {
            "test_loader": test_loader,
            "device": eval_device,
            "args": args,
        }
        self.eval_device = eval_device
        self.evaler = evaler_type(**self.eval_params)
        logger.info(
            f"Trainer: {self.__class__}, client: {client_type}, "
            f"server: {server.__class__}, evaler: {evaler_type}, num_clients: {self.num_clients}"
        )

        self.start_round = 0
        self.sweep_pruned = False
        if self.args.get('load_model_path'):
            self.load_model()

        ### best acc 
        self.best_acc = -float("inf")
        self.best_model_state = None
        self.best_epoch = -1

    def train(self):

        result_queue = mp.Queue()
        processes = []
        task_queues = []

        try:

            # set num_clients per round
            if self.num_clients <= 2000:
                M = min((5, self.num_clients))
            elif self.num_clients <= 4000:
                M = 10
            else:
                M = 10

            if self.args.multiprocessing:
                ngpus_per_node = torch.cuda.device_count()
                task_queues = [mp.Queue() for _ in range(M)]
                processes = [
                    mp.get_context('spawn').Process(
                        target=local_update,
                        args=(
                            i % ngpus_per_node,
                            task_queues[i],
                            result_queue,
                            self.args,
                            self.client_type,
                            self.model
                        )
                    )
                    for i in range(M)
                ]
                for p in processes:
                    p.start()

            if self.args.use_tqdm is True:
                epoch_iter = trange(
                    self.start_round,
                    self.global_rounds,
                    desc="FL rounds",
                    unit="round",
                )
            else:
                epoch_iter = range(self.start_round, self.global_rounds)

            enc_time = 0.0
            dec_time = 0.0
            local_train_time = 0.0
            global_agg_time = 0.0
            model_update_time = 0.0

            for epoch in epoch_iter:

                self.lr_update(epoch=epoch)

                global_state_dict = copy.deepcopy(self.model.state_dict())

                # Select clients
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
                logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

                current_lr = self.lr
                
                local_loss_dicts = defaultdict(list)
                local_deltas = defaultdict(list)

                # FedACG lookahead momentum
                if self.args.server.get('FedACG'):
                    assert self.args.server.momentum > 0
                    self.model = copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                    global_state_dict = copy.deepcopy(self.model.state_dict())


                # Client-side
                start = time.time()
                for i, client_idx in enumerate(selected_client_ids):
                    local_dataset = self.train_dataset.get_client_dataset(int(client_idx))
                    task_queue_input = {
                        'state_dict': self.model.state_dict(),
                        'client_idx': client_idx,
                        'local_dataset': local_dataset,
                        'local_lr': current_lr,
                        'global_epoch': epoch,
                    }

                    if self.args.multiprocessing:
                        task_queues[i].put(task_queue_input)

                        del task_queue_input, local_dataset
                    else:
                        task_queue = mp.Queue()
                        task_queue.put(task_queue_input)
                        local_update(self.device, task_queue, result_queue, 
                            self.args, self.client_type, self.model)

                        local_state_dict, local_loss_dict = result_queue.get()
                        for loss_key in local_loss_dict:
                            local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                        for param_key in local_state_dict:
                            local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

                if self.args.multiprocessing:
                    for _ in range(len(selected_client_ids)):
                        # Retrieve results from the queue
                        local_state_dict, local_loss_dict = result_queue.get()
                        for loss_key in local_loss_dict:
                            local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                        # If you want to save gpu memory, make sure that weights are not allocated to GPU
                        for param_key in local_state_dict:
                            local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])           

                local_train_time += time.time() - start
                logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")
                
                
                if self.args.trainer.encrypted is True:
                    # heaan setting
                    context, sk, pk, ect, dct, evt, bts, dt, log_slots = heaan_setting()
                    num_slots = 2 ** log_slots
                    
                    # encrypt client's delta
                    start = time.time()
                    
                    # flatten each client's delta
                    client_vecs = [[] for _ in range(M)]
                    param_meta = [] # for restoring the model structure

                    for param_key in global_state_dict.keys():
                        ref_tensor = global_state_dict[param_key]
                        numel = ref_tensor.numel()
                        param_meta.append((param_key, ref_tensor.shape, numel))

                        for client_idx in range(M):
                            delta_tensor = local_deltas[param_key][client_idx]
                            client_vecs[client_idx].append(delta_tensor.reshape(-1).cpu())

                    # encrypt each client's delta vector
                    encrypted_local_deltas_complex = []

                    for i, vec in enumerate(client_vecs):
                        vec = torch.cat(vec, dim=0).detach().cpu().numpy()
                        complex_vec = real_to_complex(vec, num_slots)

                        tmp = [hn.Ciphertext(context) for _ in range(math.ceil(len(complex_vec)/num_slots))]
                        enc(ect, pk, dt, complex_vec, log_slots, tmp, level=self.args.trainer.enc_level)
                        to_host(tmp)
                        encrypted_local_deltas_complex.append(tmp)
                    
                    enc_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Enc End. Total Time: {time.time() - start:.2f}s")
                    
                    weight_len = len(torch.cat(client_vecs[0], dim=0).detach().cpu().numpy())
                    
                    # Server-side
                    start = time.time()
                    ret_complex, server_stats = self.server.encrypted_aggregate(encrypted_local_deltas_complex,
                                                        selected_client_ids, weight_len)
                    global_agg_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Aggregation End. Time: {time.time() - start:.2f}s")
                    logger.info(f"Global epoch {epoch}, Aggregation End. Total Time: {global_agg_time}s")
                
                    # decrypt optimization step
                    start = time.time()

                    opt_step = np.empty(num_slots * len(ret_complex), dtype=np.complex128)
                    dec(dct, sk, ret_complex, log_slots, opt_step, num_slots * len(ret_complex), complex=True)
                    to_host(ret_complex)

                    dec_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Dec End. Total Time: {time.time() - start:.2f}s")

                    # update weight
                    opt_step_real = complex_to_real(opt_step, num_slots)
                    opt_step_real = opt_step_real[:weight_len]
                    
                    start = time.time()

                    restored = {}
                    start_id = 0

                    for param_key, shape, numel in param_meta:
                        vec = opt_step_real[start_id:start_id + numel]
                        tensor = torch.from_numpy(vec).reshape(shape)
                        tensor = tensor.to(device=global_state_dict[param_key].device, dtype=global_state_dict[param_key].dtype)
                        restored[param_key] = tensor
                        start_id += numel

                    updated_global_state_dict = copy.deepcopy(global_state_dict)
                    for param_key in global_state_dict.keys():
                        updated_global_state_dict[param_key] += restored[param_key]
                        
                    self.model.load_state_dict(updated_global_state_dict)

                    model_update_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Model update End. Total Time: {time.time() - start:.2f}s")
                
                if self.args.trainer.encrypted is False:
                    start = time.time()
                    # Server-side
                    updated_global_state_dict, server_stats = self.server.aggregate(local_deltas,
                                                                    selected_client_ids, copy.deepcopy(global_state_dict))
                    global_agg_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Aggregation End. Total Time: {time.time() - start:.2f}s")

                    start = time.time()
                    self.model.load_state_dict(updated_global_state_dict)
                    model_update_time += time.time() - start
                    logger.info(f"Global epoch {epoch}, Model update End. Total Time: {time.time() - start:.2f}s")

                wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
                wandb_dict['lr'] = self.lr

                if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                    eval_results = self.evaluate(epoch=epoch)
                    acc = eval_results["acc"]

                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        logger.warning(f"[Epoch {epoch}] New best acc updated: {acc:.2f}%")

                if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                    if self.best_model_state is not None:
                        self.save_best_model(epoch=epoch)

                stat_metrics = ["mean", "std", "min", "max", "median", "skew"]

                for stat_name in ["delta", "m", "v", "v_add_tau", "w"]:
                    if stat_name not in server_stats:
                        continue

                    for metric in stat_metrics:
                        if metric not in server_stats[stat_name]:
                            continue

                        wandb_key = f"server/{stat_name}_{metric}"
                        wandb_dict[wandb_key] = server_stats[stat_name][metric]

                self.wandb_log(wandb_dict, step=epoch)  

                gc.collect()

            return enc_time, dec_time, local_train_time, global_agg_time, model_update_time

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Terminating worker processes...")
            raise

        finally:
            if self.args.multiprocessing:
                terminate_processes(task_queues, processes)

    def lr_update(self, epoch: int) -> None:
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay ** epoch)

    def load_model(self) -> None:
        if self.args.get('load_model_path'):
            saved_dict = torch.load(self.args.load_model_path)
            self.model.load_state_dict(saved_dict['model_state_dict'], strict=False)
            self.start_round = saved_dict["epoch"] + 1
            logger.warning(f'Load model from {self.args.load_model_path}, epoch {saved_dict["epoch"]}')

    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def evaluate(self, epoch: int) -> Dict:
        results = self.evaler.eval(model=copy.deepcopy(self.model), epoch=epoch)
        acc = results["acc"]

        wandb_dict = {
            f"acc/{self.args.dataset.name}": acc,
        }

        logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')
        self.wandb_log(wandb_dict, step=epoch)

        return {"acc": acc}

    def save_best_model(self, epoch: int = -1, suffix: str = "best") -> None:
        model_path = self.exp_path / self.args.output_model_path
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch < self.args.trainer.global_rounds - 1:
            model_path = Path(f"{model_path}.e{epoch+1}")

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")

        save_dict = {
            "epoch": self.best_epoch,
            "model_state_dict": self.best_model_state,
            "best_acc": self.best_acc,
        }

        torch.save(save_dict, model_path)
        print(f"Saved best model at {model_path} (best epoch: {self.best_epoch}, best acc: {self.best_acc:.2f})")