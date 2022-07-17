Example for a Shift Happens task on ImageNet-Patch
==================================================

Evaluate the robustness of classifiers to the presence of adversarial patches 
from the ImageNet-Patch dataset [1]. 
The raw images (before application of the patches) come from the validation 
set of ImageNet, and are the same set used in RobustBench [2].

![image](https://user-images.githubusercontent.com/23276849/175907315-fb8ddf46-23c7-461f-870a-7f09638c98ae.png)


## Dataset Creation
The raw images are collected from the validation set of the ImageNet dataset. 

The ten optimized patches proposed in the paper are then applied to such images to create the perturbed data.
The patch optimization follows Brown et al.[3].


![image](https://user-images.githubusercontent.com/23276849/175907709-7bd272e1-1808-432b-a7ca-ebe083d9a5da.png)

## Evaluation Metrics

- Robust accuracy: correct classification of the images with the applied patch.

## Expected Insights/Relevance

This dataset was designed to measure the robustness of the models to the application of adversarial patches. Adding this test can give further insights on robustness of the model as a general property, beyond the evaluation that is usually performed against one specific perturbation bound (e.g., $L_\infty$ norm with maximum size of 8/255)

## Access

We release the dataset on Zenodo (https://zenodo.org/record/6568778), as well as the code to generate patches and to apply them dynamically to custom images (https://github.com/pralab/ImageNet-Patch).

## Data

The dataset is hosted on Zenodo: https://zenodo.org/record/6568778

## License

We released the data with the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).



* [1] Maura Pintor, Daniele Angioni, Angelo Sotgiu, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli:
ImageNet-Patch: A Dataset for Benchmarking Machine Learning Robustness against Adversarial Patches. CoRR abs/2203.04412 (2022)
* [2] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, Matthias Hein:
RobustBench: a standardized adversarial robustness benchmark. NeurIPS Datasets and Benchmarks 2021
* [3] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer:
Adversarial Patch. CoRR abs/1712.09665 (2017)

  