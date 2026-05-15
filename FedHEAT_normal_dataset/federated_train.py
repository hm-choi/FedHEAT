import os
from pathlib import Path

import torch
import wandb
from torch.multiprocessing import set_start_method
from datasets.build import build_datasets
from models.build import build_encoder
from servers.build import build_server
from clients.build import get_client_type
from evalers.build import get_evaler_type
from trainers.build import get_trainer_type

from utils import initalize_random_seed

import hydra
from omegaconf import DictConfig
import omegaconf
import coloredlogs, logging
import warnings
import numpy as np
import time

from servers.he_engine import heaan_setting

warnings.filterwarnings("ignore", category=np.ComplexWarning)


# import loggings
logger = logging.getLogger(__name__)
# coloredlogs.install(fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

wandb.require("service")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args : DictConfig) -> None:

    torch.multiprocessing.set_sharing_strategy('file_system')
    set_start_method('spawn', True)
    # pid = os.getpid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    args.log_dir = Path(args.log_dir)
    exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
    args.log_dir = args.log_dir / args.dataset.name / exp_name
    print(exp_name)
    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    ## Wandb
    if args.wandb:
        wandb.init(project=args.dataset.name + '_multiple_run',
                group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
                job_type=exp_name,
                dir=args.log_dir,)
        wandb.run.name = exp_name
        wandb.config.update(omegaconf.OmegaConf.to_container(
            args, resolve=True, throw_on_missing=True
        ))

    initalize_random_seed(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = build_encoder(args)
    client_type = get_client_type(args)
    server = build_server(args)
    datasets = build_datasets(args)
    evaler_type = get_evaler_type(args)

    trainer_type = get_trainer_type(args)
    trainer = trainer_type(model=model, client_type=client_type, server=server, evaler_type=evaler_type,
                           datasets=datasets,
                           device=device, args=args, config=None)
    
    logger.info("Training started.")

    if args.trainer.encrypted is True:
        # HEAAN key setting
        heaan_setting()

    start = time.time()
    enc_time, dec_time, local_train_time, global_agg_time, model_update_time = trainer.train()
    elapsed = time.time() - start

    def sec_to_hms(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}h {int(m):02d}m {s:06.3f}s"


    logger.info(
        f"\n"
        f"========== Training Finished ==========\n"
        f"Total Elapsed Time : {sec_to_hms(elapsed)} ({elapsed:.3f} sec)\n"
        f"Local Train Time   : {sec_to_hms(local_train_time)} ({local_train_time:.3f} sec)\n"
        f"Encryption Time    : {sec_to_hms(enc_time)} ({enc_time:.3f} sec)\n"
        f"Decryption Time    : {sec_to_hms(dec_time)} ({dec_time:.3f} sec)\n"
        f"Global Agg Time    : {sec_to_hms(global_agg_time)} ({global_agg_time:.3f} sec)\n"
        f"Model Update Time  : {sec_to_hms(model_update_time)} ({model_update_time:.3f} sec)\n"
        f"======================================="
    )

if __name__ == '__main__':
    main()