# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.food101.dataset import (
    Food101FeaturesDataset,
    Food101ImageDataset,
)
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("food101")
class Food101Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "food101"
        self.dataset_class = Food101ImageDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/food101/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        config = config

        if config.use_features:
            self.dataset_class = Food101FeaturesDataset

        self.dataset = super().load(config, dataset_type, *args, **kwargs)

        return self.dataset