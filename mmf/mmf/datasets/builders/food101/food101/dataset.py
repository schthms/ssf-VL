import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset

LABEL_TO_INT_MAPPING = {'nachos': 0,
                        'eggs_benedict': 1,
                        'pizza': 2,
                        'red_velvet_cake': 3,
                        'apple_pie': 4,
                        'fried_calamari': 5,
                        'sushi': 6,
                        'spaghetti_bolognese': 7,
                        'ceviche': 8,
                        'prime_rib': 9,
                        'frozen_yogurt': 10,
                        'baby_back_ribs': 11,
                        'dumplings': 12,
                        'hummus': 13,
                        'omelette': 14,
                        'huevos_rancheros': 15,
                        'fried_rice': 16,
                        'breakfast_burrito': 17,
                        'donuts': 18,
                        'ramen': 19,
                        'ravioli': 20,
                        'ice_cream': 21,
                        'deviled_eggs': 22,
                        'bibimbap': 23,
                        'pancakes': 24,
                        'scallops': 25,
                        'lasagna': 26,
                        'samosa': 27,
                        'chocolate_cake': 28,
                        'risotto': 29,
                        'hamburger': 30,
                        'onion_rings': 31,
                        'club_sandwich': 32,
                        'french_onion_soup': 33,
                        'guacamole': 34,
                        'gyoza': 35,
                        'poutine': 36,
                        'creme_brulee': 37,
                        'strawberry_shortcake': 38,
                        'caesar_salad': 39,
                        'carrot_cake': 40,
                        'spring_rolls': 41,
                        'waffles': 42,
                        'miso_soup': 43,
                        'chocolate_mousse': 44,
                        'shrimp_and_grits': 45,
                        'pho': 46,
                        'baklava': 47,
                        'paella': 48,
                        'foie_gras': 49,
                        'hot_dog': 50,
                        'greek_salad': 51,
                        'chicken_wings': 52,
                        'lobster_bisque': 53,
                        'cup_cakes': 54,
                        'edamame': 55,
                        'gnocchi': 56,
                        'bread_pudding': 57,
                        'french_fries': 58,
                        'cheesecake': 59,
                        'steak': 60,
                        'pad_thai': 61,
                        'bruschetta': 62,
                        'tuna_tartare': 63,
                        'pulled_pork_sandwich': 64,
                        'chicken_curry': 65,
                        'falafel': 66,
                        'caprese_salad': 67,
                        'tiramisu': 68,
                        'tacos': 69,
                        'grilled_cheese_sandwich': 70,
                        'seaweed_salad': 71,
                        'sashimi': 72,
                        'croque_madame': 73,
                        'churros': 74,
                        'lobster_roll_sandwich': 75,
                        'macaroni_and_cheese': 76,
                        'fish_and_chips': 77,
                        'mussels': 78,
                        'filet_mignon': 79,
                        'peking_duck': 80,
                        'cheese_plate': 81,
                        'spaghetti_carbonara': 82,
                        'pork_chop': 83,
                        'panna_cotta': 84,
                        'cannoli': 85,
                        'garlic_bread': 86,
                        'clam_chowder': 87,
                        'grilled_salmon': 88,
                        'takoyaki': 89,
                        'crab_cakes': 90,
                        'french_toast': 91,
                        'beef_tartare': 92,
                        'hot_and_sour_soup': 93,
                        'beef_carpaccio': 94,
                        'beignets': 95,
                        'escargots': 96,
                        'beet_salad': 97,
                        'chicken_quesadilla': 98,
                        'oysters': 99,
                        'macarons': 100}


class Food101FeaturesDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "food101", config, dataset_type, imdb_file_index, *args, **kwargs
        )
        assert (
            self._use_features
        ), "config's 'use_features' must be true to use feature dataset"

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        recipe = sample_info["text"]
        if isinstance(recipe, list):
            recipe = recipe[0]
        processed_sentence = self.text_processor({"text": recipe})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_features is True:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        if isinstance(sample_info["label"], list):
            label = sample_info["label"][0]

        label = LABEL_TO_INT_MAPPING[label]
        current_sample.targets = torch.tensor(label, dtype=torch.long)

        return current_sample


class Food101ImageDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "food101", config, dataset_type, imdb_file_index, *args, **kwargs
        )
        assert (
            self._use_images
        ), "config's 'use_images' must be true to use image dataset"

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        recipe = sample_info["text"]
        if isinstance(recipe, list):
            recipe = recipe[0]
        processed_sentence = self.text_processor({"text": recipe})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_images is True:
            current_sample.image = self.image_db[idx]["images"][0]

        if isinstance(sample_info["label"], list):
            label = sample_info["label"][0]

        label = LABEL_TO_INT_MAPPING[label]
        current_sample.targets = torch.tensor(label, dtype=torch.long)

        return current_sample
