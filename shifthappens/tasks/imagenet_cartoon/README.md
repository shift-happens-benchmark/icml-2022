# Task Description
ImageNet-Cartoon: a dataset to benchmark the robustness of ImageNet models against domain shifts.

<img width="1029" alt="examples" src="https://user-images.githubusercontent.com/38894497/178097450-8b888907-d2b9-46f0-bfb7-c99700a92b1e.png">
Several examples of ImageNet images (top) and their respective ImageNet-Cartoon (middle) and ImageNetDrawing (bottom) versions.

## Dataset Creation
Images are taken from the ImageNet dataset and then transformed into cartoons using the GAN framework proposed by [1].

[1] Wang, X. and Yu, J. Learning to cartoonize using white-box cartoon representations. In _Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR)_, June 2020.

## Evaluation Metrics
Robust accuracy: correct classification of the transformed images.

## Expected Insights/Relevance
The accuracy of pretrained ImageNet models decreases significantly on the proposed dataset.

## Access
We release the dataset on Zenodo (https://zenodo.org/record/6801109), as well as the code to generate it  (https://github.com/oberman-lab/imagenet-shift).

## Data
The dataset is hosted on Zenodo (https://zenodo.org/record/6801109).

## License
We released the data with the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
