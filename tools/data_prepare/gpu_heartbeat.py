#!/usr/bin/env python3
"""Keep allocated GPUs visibly active during CPU-heavy setup/download phases."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="gpu-heartbeat")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--tensor-mb", type=float, default=4.0)
    parser.add_argument("--all-devices", action="store_true")
    parser.add_argument("--stop-file", default="", help="Exit gracefully when this file appears.")
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
    stop_file = Path(args.stop_file).expanduser() if args.stop_file else None

    def should_stop() -> bool:
        return stop or (stop_file is not None and stop_file.exists())

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
    while not should_stop():
        tick += 1
        for device_id, tensor in zip(device_ids, tensors):
            with torch.cuda.device(device_id):
                tensor.add_(1.0)
                if tick % 128 == 0:
                    tensor.fill_(1.0)
                torch.cuda.synchronize(device_id)
        sleep_remaining = max(1.0, float(args.interval))
        while sleep_remaining > 0 and not should_stop():
            sleep_step = min(1.0, sleep_remaining)
            time.sleep(sleep_step)
            sleep_remaining -= sleep_step

    print(f"[{args.label}] GPU heartbeat stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
