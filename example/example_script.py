"""Dummy script that prints to stdout + stderr, and allocates some memory."""

import sys
import time

import torch
import tyro


def main(num: int) -> None:
    assert torch.cuda.is_available()

    tensors = []

    for i in range(num):
        print(f"Hello, world #{i}", flush=True)
        print(f"Error log #{i}", file=sys.stderr)
        tensors.append(torch.randn((4096, 4096)).cuda())
        time.sleep(0.5)


if __name__ == "__main__":
    tyro.cli(main)
