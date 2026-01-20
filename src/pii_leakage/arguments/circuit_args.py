# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field


@dataclass
class CircuitArgs:
    CONFIG_KEY = "circuit_args"

    ablation: str = field(
        default="zero",
        metadata={
            "help": "circuit ablation method",
        },
    )

    patch: str = field(
        default="self",
        metadata={
            "help": "circuit patching method",
        },
    )

    threshold: float = field(
        default=0.000077,
        metadata={"help": "circuit score threshold for selecting PII nodes."},
    )

    scale_factor: float = field(
        default=0.1,
        metadata={"help": "circuit scale factor for adjusting attention scores."},
    )
