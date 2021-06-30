# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.food101.masked_dataset import MaskedFood101Dataset
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("masked_food101")
class MaskedFood101Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_food101"
        self.dataset_class = MaskedFood101Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/food101/masked.yaml"
