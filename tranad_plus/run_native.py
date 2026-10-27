
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from config import BASE_DIR, NATIVE_PACKAGE, log_directory


def check_device(device, parser):
    import torch

    if device == "cpu":
        return

    if device == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS is not available.")
        return

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            parser.error("CUDA is not available.")

        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            parser.error(f"Invalid device: {device}")

        if not 0 <= index < torch.cuda.device_count():
            parser.error(f"Unavailable CUDA device: {device}")

        return

    parser.error(f"Unsupported device: {device}")


def native_command(script, options):
    path = NATIVE_PACKAGE / "scripts" / script

    if not path.is_file():
        raise FileNotFoundError(
            f"Native script not found: {path}"
        )

    return [
        sys.executable,
        "-u",
        str(path),
        *options,
    ]


def create_run_id():
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    return f"{timestamp}_{uuid4().hex[:8]}"


def run_and_log(command, dataset, identifier, stage):
    if stage not in ("train", "evaluate"):
        raise ValueError(
            f"Unsupported execution stage: {stage}"
        )

    directory = log_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    run_id = create_run_id()

    destination = directory / (
        f"{identifier}_{stage}_{run_id}.log"
    )

    environment = os.environ.copy()

    if stage == "evaluate":
        environment[
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ] = "1"

    print(
        f"Command: {' '.join(map(str, command))}",
        flush=True,
    )

    print(f"Log: {destination}", flush=True)

    with destination.open(
        "x",
        encoding="utf-8",
    ) as log:
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )

        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()

            returncode = process.wait()

        except KeyboardInterrupt:
            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            raise

        finally:
            process.stdout.close()

    if returncode < 0:
        number = -returncode

        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "UNKNOWN"

        raise RuntimeError(
            f"Process terminated by {name} "
            f"(signal {number}). Log: {destination}"
        )

    if returncode != 0:
        raise RuntimeError(
            f"Process failed with exit code {returncode}. "
            f"Log: {destination}"
        )

    print(
        f"{stage.capitalize()} completed. "
        f"Log: {destination}",
        flush=True,
    )

    return destination
