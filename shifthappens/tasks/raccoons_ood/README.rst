Example for a Shift Happens task on Raccoons dataset
====================================================
This task aims to evaluate models' out-of-distribution (OOD) detection on 200 raccoon images.
Raccoons are not presented in ImageNet classes, so the task uses models'
confidences (maximal predicted class probability) for the ImageNet validation set and
raccoons images (ImageNet samples treated as class 1 and raccoons as class 0) to measure
AUROC and FPR at TPR equal 0.95.

The original dataset was collected by Dat Tran for the object detection task
and can be found at https://github.com/datitran/raccoon_dataset.
