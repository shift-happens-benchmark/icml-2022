# Task Description

We introduce a set of image transformations that can be used as corruptions to evaluate the robustness of models. The primary distinction of the proposed transformations is that, unlike existing approaches such as Common Corruptions [[1](https://arxiv.org/abs/1903.12261)], the geometry of the scene is incorporated in the transformations – thus leading to corruptions that are more likely to occur in the real world. We apply these corruptions to the ImageNet validation set to create 3D Common Corruptions (ImageNet-3DCC) benchmark. Some examples are shown below.

This benchmark is based on our CVPR'22 paper [[3](https://arxiv.org/abs/2203.01441)], visit our [project page](https://3dcommoncorruptions.epfl.ch/) and [github](https://github.com/EPFL-VILAB/3DCommonCorruptions) for more details.

![pull_imagenet](https://user-images.githubusercontent.com/9682590/175020233-e303bdd4-7e4f-4eab-a745-dff6a2b26780.jpg)

 

## Dataset Creation

Most of the corruptions require an RGB image and scene depth, except the noise ones that can be generated from RGB image directly. As ImageNet images do not have depth labels, we generate depth predictions using a state-of-the-art depth estimator [[2](https://arxiv.org/abs/2110.04994),[3](https://arxiv.org/abs/2203.01441)]. We described the methods used to generate the corruptions in our [ShiftHappens paper](https://openreview.net/pdf?id=Evar7nqAQtL) and in [[3](https://arxiv.org/abs/2203.01441)]. 

## Evaluation Metrics

We use the mean Corruption Error as defined in [[1](https://arxiv.org/abs/1903.12261)]. It is relevant as we also defined five levels of shift intensities for each corruption.

## Expected Insights/Relevance

3DCC incorporates 3D information to generate corruptions that are consistent with the scene geometry. This leads to shifts that are more likely to occur in the real world. We expect to gain insights to model vulnerabilities under real-world plausible corruptions that are not captured by 2D corruptions.

# Access

## Data

Instructions for downloading the entire dataset can be found [here](https://github.com/EPFL-VILAB/3DCommonCorruptions#3dcc-data). Alternatively, to download individual corruptions, the link is given by `https://datasets.epfl.ch/3dcc/imagenet_3dcc/x.tar.gz` where x is the name of the corruption i.e. 
`x=[near_focus,far_focus,fog_3d,flash,color_quant,low_light,xy_motion_blur,z_motion_blur,iso_noise,bit_error,h265_abr,h265_crf]`.


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/EPFL-VILAB/3DCommonCorruptions/blob/main/LICENSE) for details.

----
[[1](https://arxiv.org/abs/1903.12261)] Hendrycks et al., Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, ICLR'19.

[[2](https://arxiv.org/abs/2110.04994)] Eftekhar et al., Omnidata: A Scalable Pipeline for Making Multi-task Mid-level Vision Datasets from 3D Scans, ICCV'21.

[[3](https://arxiv.org/abs/2203.01441)] Kar et al., 3D Common Corruptions and Data Augmentation, CVPR'22.


## Citation
If you find the code, or data useful, please cite these papers:

```
@inproceedings{kar20223d_shifthappens,
  title={3D Common Corruptions for Object Recognition},
  author={Kar, Oguzhan Fatih and Yeo, Teresa and Zamir, Amir},
  booktitle={ICML 2022 Shift Happens Workshop}
}
@inproceedings{kar20223d,
  title={3D Common Corruptions and Data Augmentation},
  author={Kar, O{\u{g}}uzhan Fatih and Yeo, Teresa and Atanov, Andrei and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18963--18974},
  year={2022}
}
```