"""Dummy script that prints to stdout + stderr, and allocates some memory."""

import sys
import time

import torch
import tyro


def main(num: int) -> None:
    assert torch.cuda.is_available()
    
    # Print GPU information
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs", flush=True)
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {device_name}", flush=True)

    tensors = []

    for i in range(num):
        print(f"Hello, world #{i}", flush=True)
        print(f"Error log #{i}", file=sys.stderr)
        tensors.append(torch.randn((4096, 4096)).cuda())
        time.sleep(0.5)

    assert num % 2 == 0


if __name__ == "__main__":
    tyro.cli(main)
