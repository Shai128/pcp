import re
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from enum import Enum
from enum import auto


class CorruptionType(Enum):
    MISSING_Y = 'missing_y'
    MISSING_X = 'missing_x'
    NOISED_X = 'noised_x'
    NOISED_Y = 'noised_y'
    DISPERSIVE_NOISED_Y = 'dispersive_noised_y'


corruption_type_string_to_enum = {
    'missing_y': CorruptionType.MISSING_Y,
    'missing_x': CorruptionType.MISSING_X,
    'noised_x': CorruptionType.NOISED_X,
    'noised_y': CorruptionType.NOISED_Y,
    'dispersive_noised_y': CorruptionType.DISPERSIVE_NOISED_Y,
}


def get_corruption_type_from_dataset_name(dataset_name: str) -> CorruptionType:
    matches = get_corruption_types_from_dataset_name(dataset_name)
    if len(matches) != 1:
        print(f"warning: invalid dataset name: {dataset_name}, has {len(matches)} corruption types instead of 1")
        return None
    corruption_type_string = matches[0]
    return corruption_type_string_to_enum[corruption_type_string]


def get_corruption_types_from_dataset_name(dataset_name: str) -> List[str]:
    corruption_choices = list(corruption_type_string_to_enum.keys())
    pattern = '|'.join(map(re.escape, corruption_choices))
    matches = re.findall(pattern, dataset_name)
    return matches


def remove_corruption_type_from_dataset_name(dataset_name: str) -> str:
    matches = get_corruption_types_from_dataset_name(dataset_name)
    for match in matches:
        dataset_name = dataset_name.replace(match+"_", "")
    return dataset_name
