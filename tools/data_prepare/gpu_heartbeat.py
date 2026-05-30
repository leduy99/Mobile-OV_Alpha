#!/usr/bin/env python3
"""Keep allocated GPUs visibly active during CPU-heavy setup/download phases."""

from __future__ import annotations

import argparse
import math
import signal
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="gpu-heartbeat")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--tensor-mb", type=float, default=4.0)
    parser.add_argument("--all-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print(f"[{args.label}] CUDA is not available; heartbeat disabled.", flush=True)
        return 0

    stop = False

    def _stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    device_count = torch.cuda.device_count()
    device_ids = list(range(device_count)) if args.all_devices else [0]
    elems = max(1024, int((float(args.tensor_mb) * 1024 * 1024) / 4))
    side = max(32, int(math.sqrt(elems)))
    tensors = []
    for device_id in device_ids:
        with torch.cuda.device(device_id):
            tensors.append(torch.ones((side, side), device=f"cuda:{device_id}", dtype=torch.float32))
            torch.cuda.synchronize(device_id)

    print(
        f"[{args.label}] GPU heartbeat active on devices={device_ids}, "
        f"tensor_shape={side}x{side}, interval={args.interval}s",
        flush=True,
    )

    tick = 0
    while not stop:
        tick += 1
        for device_id, tensor in zip(device_ids, tensors):
            with torch.cuda.device(device_id):
                tensor.add_(1.0)
                if tick % 128 == 0:
                    tensor.fill_(1.0)
                torch.cuda.synchronize(device_id)
        time.sleep(max(1.0, float(args.interval)))

    print(f"[{args.label}] GPU heartbeat stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
