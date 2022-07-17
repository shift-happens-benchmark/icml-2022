# Semantic Shift Benchmark

We propose the Semantic Shift Benchmark suite (SSB) for open-set recognition and related tasks (e.g category discovery, out-of-distribution detection etc.). The SSB is intended to isolate semantic novelty from other forms of distributional shifts, and consists of three fine-grained datasets and a large-scale ImageNet benchmark.

For each dataset, we provide 'seen' and 'unseen' class splits. Furthermore, for each dataset, the 'unseen' classes are divided into 'Easy' and 'Hard'. Harder examples are more visually and semantically similar to the 'seen' classes.

## ImageNet Benchmark

The ImageNet evaluation is implemented here. To run, first: 
* Download and process ImageNet-21K-P, [details for which can be found here](https://github.com/Alibaba-MIIL/ImageNet21K). 
* Set paths to ImageNet-21K-P and the regular ImageNet dataset in ```shifthappens/tasks/ssb/imagenet_ssb.py``` (in ```SSB.resource```) and ```shifthappens/data/imagenet.py``` (in ```ImageNetValidationData```)

Finally, run:

```
import shifthappens

from shifthappens.models.torchvision import ResNet18
from shifthappens import benchmark as sh_benchmark

from shifthappens.tasks.ssb import semantic_shift_benchmark

sh_benchmark.evaluate_model(
    ResNet18(device="cpu", max_batch_size=128),
    "data",
)
```

## Examples

For more example from other evaluations in the Semantic Shift Benchmark, see [here](https://www.robots.ox.ac.uk/~vgg/research/osr/#ssb_suite).

![image](assets/ssb_imagenet.png)

## Citation

If you find this useful, please consider citing our paper:
```
@InProceedings{vaze2022openset,
               title={Open-Set Recognition: a Good Closed-Set Classifier is All You Need?},
               author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
               booktitle={International Conference on Learning Representations},
               year={2022}}
```