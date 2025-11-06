'''
Main module for post-training tasks at Uniandes.

Input will be the .yaml configuration file.
'''


import argparse
import yaml
from pathlib import Path
import logging
import os
import multiprocessing as mp
from postraining_uniandes.logger_helper import get_logger
import pprint
from postraining_uniandes.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], #Fixed for 4 GPUs at our server. 
    help="which devices to use on local machine",
)


def main_process(rank, fname, num_gpus, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    logger = get_logger(force=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine) -> Must check if needed. For now, singular GPU is gonna be used.
    #world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    #logger.info(f"Running... (rank: {rank}/{world_size})")

    train(params)
    

if __name__ == "__main__":
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method("spawn")
    for rank in range(num_gpus): #In case of multiple GPU usage. 
        mp.Process(target=main_process, args=(rank, args.fname, num_gpus, args.devices)).start()