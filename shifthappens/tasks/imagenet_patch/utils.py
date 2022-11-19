"""
Utilities for ImageNet Patch.
"""
import os.path

from torchvision import datasets


class ImageFolderEmptyDirs(datasets.ImageFolder):
    """
    This is required for handling empty folders from the ImageFolder Class.
    """

    def find_classes(self, directory):
        """Remaps the empty folders to actual classes in imagenet."""
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(classes)
            if len(os.listdir(os.path.join(directory, cls_name))) > 0
        }
        return classes, class_to_idx
