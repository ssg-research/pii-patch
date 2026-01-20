# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass, field


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"

    num_workers: int = field(default=2, metadata={
        "help": "number of workers"
    })

    log_every: int = field(default=100, metadata={
        "help": "log interval for training"
    })

    save_every: int = field(default=249, metadata={
        "help": "save interval for training"
    })

    device: str = field(default="cuda", metadata={
        "help": "device to run observers on"
    })

    batch_size: int = field(default=64, metadata={
        "help": "default batch size for training"
    })

    eval_batch_size: int = field(default=32, metadata={
        "help": "default batch size for inference"
    })

    verbose: bool = field(default=True, metadata={
        "help": "whether to print out to the cmd line"
    })

    skip_ppl_eval: bool = field(default=False, metadata={
        "help": "whether to skip ppl eval"
    })

    use_multi_gpu: bool = field(default=None, metadata={
        "help": "whether to use multi-GPU training with DataParallel. None=auto-detect, True=force multi-GPU, False=force single GPU"
    })

    force_single_gpu: bool = field(default=False, metadata={
        "help": "force single GPU usage even when multiple GPUs are available (overrides use_multi_gpu)"
    })
