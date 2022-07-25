## Task Description
ImageNet-Drawing: a dataset to benchmark the robustness of ImageNet models against domain shifts.

## Dataset Creation
Images are taken from the ImageNet dataset and then transformed into drawings using the simple image processing described in [1].

[1] Lu, C., Xu, L., and Jia, J. Combining Sketch and Tone for Pencil Drawing Production. In Asente, P. and Grimm, C.
(eds.), _International Symposium on Non-Photorealistic Animation and Rendering_. The Eurographics Association, 2012. ISBN 978-3-905673-90-6. doi: 10.2312/PE/NPAR/NPAR12/065-073.

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
