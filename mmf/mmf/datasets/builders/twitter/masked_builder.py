# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.twitter.masked_dataset import MaskedTwitterDataset
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("masked_twitter")
class MaskedTwitterBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_twitter"
        self.dataset_class = MaskedTwitterDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/twitter/masked.yaml"
