# SI-SCORE

([Dataset homepage](https://github.com/google-research/si-score))

# Task Description

As we begin to bring image classification models into the real world, it is important that they are robust and do not overfit to academic datasets. While robustness is hard to define, one can think of intuitive factors that image classification models should be robust to, such as changes in object location or differing weather conditions. Models should at least be robust to these factors. To test this, ideally we would have datasets that vary only these factors. For some factors such as image compression and differing weather conditions, datasets such as Imagenet-C already exist, whereas for common-sense factors such as object size and object location, such datasets did not yet exist prior to SI-SCORE.

In SI-SCORE, we take objects and backgrounds and systematically vary object size, location and rotation angle so we can study the effect of changing these factors on model performance. We also provide the code, object images and background images and hope that researchers can create their own datasets to test robustness to other factors of variation.

Here are some sample images from the dataset:

<img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_default_big_2.jpg" width="200"> <img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_loc_23_2.jpg" width="200"> <img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_rotate_50_2.jpg" width="200">

<img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_default.jpg" width="200"> <img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_loc_76.jpg" width="200"> <img src="https://raw.githubusercontent.com/google-research/si-score/master/sample_images/demo_img_koala_rotate_230.jpg" width="200">

## Dataset Creation


## Evaluation Metrics
The dataset uses accuracy as an evaluation metric. One can also calculate the accuracy per e.g. object size to compare model performance across different object sizes (/locations of the object in the image / object rotation angles).

## Expected Insights/Relevance
Discovering differences in robustness to changes in object size/location/rotation angle between image classification models, such as ResNet-50s with GroupNorm being better at classifying small objects than ResNet-50s with BatchNorm. (See [our workshop paper](https://arxiv.org/abs/2104.04191) for more details.)

Testing hypotheses such as whether Vision Transformer models perform worse when objects are on patch intersections.

# Access

## Data

[Dataset homepage](https://github.com/google-research/si-score)
Big zip files: `https://s3.us-east-1.amazonaws.com/si-score-dataset/{rotation|location|area}.zip`

## License
Apache 2.0 license.