
Example for a Shift Happens task on ImageNet-Patch
==================================================

Evaluate the robustness of classifiers to the presence of naturally corrupted patches [1]. 


Dataset Creation
----------------

The raw images are taken from the validation set of the ImageNet dataset [2].

We apply the corruption functions in [3] to corrupt every image, and replace a patch of each clean image with the corresponding corrupted patch.


Evaluation Metrics
------------------

1) Standard Classification Accuracy: the accuracy on raw images of ImageNet-1k validation dataset
2) Fooling rate: the percentrage of misclassified images 

Expected Insights/Relevance
---------------------------

This dataset was designed to measure the robustness of the models to naturally corrupted patches. This is important, especifcally for patch-based neural network architecture, e.g., Vision Transformer.

Access
------

We release the dataset on Zenodo (https://zenodo.org/record/7378906#.Y-Fhi9J_pH4).

Data
----

The dataset is hosted on Zenodo: https://zenodo.org/record/7378906

License
-------

We released the data with the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).


* [1] Jindong Gu, Volker Tresp, and Yao Qin, Evaluating Model Robustness to Patch Perturbations, ICML 2022 Shift Happens Workshop, 2022
* [2] Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li, Imagenet: A large-scale hierarchical image database, IEEE conference on computer vision and pattern recognition, 2009
* [3] Hendrycks, Dan and Dietterich, Thomas, Benchmarking neural network robustness to common corruptions and perturbations, ICLR, 2019
