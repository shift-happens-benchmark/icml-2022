"""A task for evaluating the classification accuracy on ImageNet-MetaShift.

MetaShift is constructed from the natural heterogeneity of Visual Genome and its annotations.
To support evaluating ImageNet trained models on MetaShift,
we match MetaShift classes with ImageNet hierarchy using WordNet.
The ImageNet-matched-MetaShift is a collection of 5,040 sets of images from 261 classes,
where all the labels are a subset of the ImageNet-1k.
Each class in the ImageNet-matched Metashift contains 2301.6 images on average,
and 19.3 subsets capturing images in different contexts.

"""

import dataclasses
import json
import os
import pickle
import pprint
import shutil
from collections import Counter
from collections import defaultdict

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.datasets.utils as tv_utils
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult

"""Use False to generate the MetaDataset for all ~200 classes [Warning: Very Large].
"""
ONLY_SELECTED_CLASSES = True

"""check class_info.txt to select classes
"""
SELECTED_CLASSES = ["airplane", "elephant", "cat", "dog", "horse"]

"""
change threshold to set the minimum number of images per subset
"""
IMGAGE_SUBSET_SIZE_THRESHOLD = 25

CLASS_HIERARCHY_JSON = (
    "https://zenodo.org/record/6804766/files/class_hierarchy.json?download=1"
)
CLASS_INFO_TXT = "https://zenodo.org/record/6804766/files/class_info.txt?download=1"
IMAGENET_HIERARCHY_JSON = (
    "https://zenodo.org/record/6804766/files/imagenet1k_node_names.json?download=1"
)
SELECTED_SUBSET_PKL = (
    "https://zenodo.org/record/6804766/files/selected-candidate-subsets.pkl?download=1"
)


@sh_benchmark.register_task(
    name="ImageNet-MetaShift",
    relative_data_folder="imagenet_metashift",
    standalone=True,
)
@dataclasses.dataclass
class ImageNetMetaShift(Task):
    """Evaluate the classification accuracy on a small subset of the ImageNet-MetaShift dataset.
    Each class in the test case contains several subsets, representing distribution of images in different contexts.

    [1] We have matched the labels in ImageNet-1k to MetaShift. Since ImageNet-1k has heterogeneous hierarchy,
        a class can have many breeds. Take dog as an example, MetaShift only contains one class of dog,
        while ImageNet has many kinds of dogs. In our metrics, any results under dog hierarchy are viewed as correct
        when evaluate the classification results of dog.
    """

    resources = [
        (
            "images.zip",
            "https://nlp.stanford.edu/data/gqa/images.zip",
        )
    ]

    def parse_node_str(self, node_str):
        """parse the node into class and context string"""
        tag = node_str.split("(")[-1][:-1]
        subject_str = node_str.split("(")[0].strip()
        return subject_str, tag

    def load_candidate_subsets(self):
        """
        load candidate subsets for generating
        """
        pkl_save_path = os.path.join(
            self.data_root, "imagenet-metashift", "selected-candidate-subsets.pkl"
        )
        with open(pkl_save_path, "rb") as pkl_f:
            load_data = pickle.load(pkl_f)
            print("pickle load", len(load_data), pkl_save_path)
            return load_data

    def copy_image_for_subject(
        self, root_folder, subject_str, subject_data, node_name_to_img_id, trainsg_dupes
    ):
        """Copy Image Sets: Work at subject_str level"""

        # Iterate all the subsets of the given subject
        for node_name in subject_data:
            subject_str, tag = self.parse_node_str(node_name)

            # Create dataset a new folder
            subject_localgroup_folder = os.path.join(
                root_folder, subject_str, node_name
            )
            if os.path.isdir(subject_localgroup_folder):
                shutil.rmtree(subject_localgroup_folder)
            os.makedirs(subject_localgroup_folder, exist_ok=False)

            for image_idx_in_set, imageID in enumerate(
                node_name_to_img_id[node_name] - trainsg_dupes
            ):

                src_image_path = os.path.join(
                    self.data_root,
                    "imagenet-metashift",
                    "allImages",
                    "images",
                    imageID + ".jpg",
                )
                dst_image_path = os.path.join(
                    subject_localgroup_folder, imageID + ".jpg"
                )
                shutil.copyfile(src_image_path, dst_image_path)

        return

    def preprocess_groups(self, subject_classes=SELECTED_CLASSES):
        """preprocess for each group:

        load condidate subsets for each group
        discard an object class if it has too few local groups
        copy the image files from the general dataset to to the desired groups
        """

        trainsg_dupes = set()

        # Consult back to this dict for concrete image IDs.
        node_name_to_img_id = self.load_candidate_subsets()

        # Build a default counter first
        # Data Iteration
        group_name_counter = Counter()
        for node_name in node_name_to_img_id.keys():
            # Apply a threshold: e.g., 25
            imageID_set = node_name_to_img_id[node_name]
            imageID_set = imageID_set - trainsg_dupes
            node_name_to_img_id[node_name] = imageID_set
            if len(imageID_set) >= IMGAGE_SUBSET_SIZE_THRESHOLD:
                group_name_counter[node_name] = len(imageID_set)
            else:
                pass

        most_common_list = group_name_counter.most_common()
        most_common_list = [(x, count) for x, count in group_name_counter.items()]

        # Build a subject dict
        subject_group_summary_dict = defaultdict(Counter)
        for node_name, imageID_set_len in most_common_list:
            subject_str, tag = self.parse_node_str(node_name)

            if ONLY_SELECTED_CLASSES and subject_str not in subject_classes:
                continue

            subject_group_summary_dict[subject_str][node_name] = imageID_set_len

        subject_group_summary_list = sorted(
            subject_group_summary_dict.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )

        new_subject_group_summary_list = list()
        subjects_to_all_set = defaultdict(set)

        # Subject filtering for dataset generation
        for subject_str, subject_data in subject_group_summary_list:

            # Discard an object class if it has too few local groups
            if len(subject_data) <= 10:
                continue
            else:
                new_subject_group_summary_list.append((subject_str, subject_data))

            self.copy_image_for_subject(
                self.META_DATASET_FOLDER,
                subject_str,
                subject_data,
                node_name_to_img_id,
                trainsg_dupes,
            )  # use False to share

            for node_name in subject_data:
                subject_str, tag = self.parse_node_str(node_name)
                subjects_to_all_set[subject_str].update(node_name_to_img_id[node_name])

        pprint.pprint(new_subject_group_summary_list)

        print("Done! Please check ", self.META_DATASET_FOLDER)

        return (
            node_name_to_img_id,
            most_common_list,
            subjects_to_all_set,
            subject_group_summary_dict,
        )

    def iterate_find_index(self, dt, idx=[]):
        """find all the breeds of a class"""

        if isinstance(dt, list):
            for i in dt:
                idx = self.iterate_find_index(i, idx)
        elif isinstance(dt, dict):
            if "children" not in dt:
                idx.append(dt["index"])
            else:
                idx = self.iterate_find_index(dt["children"], idx)
        idx.sort()
        return idx

    def find_imagenet_node(self, name, dt, idx=[]):
        """match the class to imagenet label"""

        if isinstance(dt, list):
            for i in dt:
                idx = self.find_imagenet_node(name, i, idx)
        elif isinstance(dt, dict):
            dict_name = dt["name"].split(",")
            if name in dict_name:
                idx = self.iterate_find_index(dt, idx)
            elif "children" in dt:
                idx = self.find_imagenet_node(name, dt["children"], idx)
        return idx

    def setup(self):
        """load the information files and generate the seleteced classes"""

        dataset_folder = os.path.join(self.data_root, "imagenet-metashift")
        # the class hierarchy information of GQA
        tv_utils.download_url(
            CLASS_HIERARCHY_JSON,
            dataset_folder,
            "class_hierarchy.json",
            "3914cd4fb68245fb65fd64818a700169",
        )
        # the selected class information of ImageNet-matched-Metashift
        tv_utils.download_url(
            CLASS_INFO_TXT,
            dataset_folder,
            "class_info.txt",
            "9f604c19c7ce09a93185b24cf84544a0",
        )
        # the class hierarchy information of ImageNet-1k
        tv_utils.download_url(
            IMAGENET_HIERARCHY_JSON,
            dataset_folder,
            "imagenet1k_node_names.json",
            "82db9c537e7887c66d19383e2c06ef12",
        )
        # the selected subsets information of ImageNet-matched-Metashift
        tv_utils.download_url(
            SELECTED_SUBSET_PKL,
            dataset_folder,
            "selected-candidate-subsets.pkl",
            "e09699b9f39b2c3000103a9cf6847c80",
        )

        # Download the pre-processed and cleaned version of Visual Genome by GQA.
        if not os.path.exists(os.path.join(dataset_folder, "allImages", "images")):
            for file_name, url in self.resources:
                sh_utils.download_and_extract_archive(
                    url, os.path.join(dataset_folder, "allImages"), None, file_name
                )

        self.META_DATASET_FOLDER = os.path.join(dataset_folder, "generated")
        if os.path.exists(self.META_DATASET_FOLDER):
            shutil.rmtree(self.META_DATASET_FOLDER)

        self.preprocess_groups()

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = tv_datasets.ImageFolder(
            root=self.META_DATASET_FOLDER, transform=test_transform
        )

        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

        self.imagenet_label_data = json.load(
            open(
                os.path.join(
                    self.data_root, "imagenet-metashift", "imagenet1k_node_names.json"
                )
            )
        )
        self.imagenet_label_data = self.imagenet_label_data["children"]

        # generate the imagenet label to class mapping
        self.imagenet_labels = {}
        for i in SELECTED_CLASSES:
            self.imagenet_labels[i] = self.find_imagenet_node(
                i, self.imagenet_label_data
            )

    def _prepare_dataloader(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluate the overall accuracy on the selected test case.
        Remove the comments of print command to see the accuracy result on each class.
        """

        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True, confidences=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)

        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        correct = 0
        for i in range(len(self.ch_dataset.classes)):
            class_name = self.ch_dataset.classes[i]
            # print("class:", class_name)
            index = np.where(np.array(self.ch_dataset.targets) == i)
            imagenet_label = np.array(self.imagenet_labels[class_name])
            imagenet_label = np.expand_dims(imagenet_label, 0)
            predict_label = all_predicted_labels[index]
            predict_label = np.expand_dims(predict_label, -1)
            predict_correct = sum(np.any((predict_label == imagenet_label), axis=-1))
            # print(f"predict_correct:{predict_correct}, total:{len(predict_label)}")
            # print(f"predict_accuracy:{predict_correct / len(predict_label)}")
            correct += predict_correct

        accuracy = correct / len(self.ch_dataset.targets)

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )
