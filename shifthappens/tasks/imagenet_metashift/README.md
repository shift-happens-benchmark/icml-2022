# Task Description
MetaShift is a collection of 12,868 sets of natural images from 410 classes. Each set corresponds to images in a similar context and represents a coherent real-world data distribution, as shown in the figure below. Different from and complementary to other efforts to curate benchmarks for data shifts, MetaShift pulls together data across different experiments or sources. It leverages heterogeneity within the large sets of images from the Visual Genome project  by clustering the images using metadata that describes the context of each image. 
To support evaluating ImageNet trained models on MetaShift, we match MetaShift with ImageNet hierarchy. The matched version covers 867 out of 1,000 classes in ImageNet-1k. Each class in the ImageNet-matched Metashift contains 2301.6 images on average, and 19.3 subsets capturing images in different contexts.
![MetaShift-Examples](https://user-images.githubusercontent.com/67904087/177720791-4c837f25-abb0-48ed-a8af-b611c3c1612f.jpg)

## Dataset Creation
We leverage the natural heterogeneity of [Visual Genome](https://visualgenome.org) and its annotations to construct MetaShift. The key construction idea is to cluster images using its metadata, which provides context for each image (e.g. _cats with cars_ or _cats in bathroom_) that represent distinct data distributions.

## Evaluation Metrics
We have matched the labels in ImageNet-1k to MetaShift. Since ImageNet-1k has heterogeneous hierarchy, a class can have many breeds. Take dog as an example, MetaShift only contains one class of dog, while ImageNet has many kinds of dogs. In our metrics, any results under dog hierarchy are viewed as correct when evaluate the classification results of dog.


## Expected Insights/Relevance
MetaShift has two important benefits: first, it contains orders of magnitude more natural data shifts than previously available. Second, it provides explicit explanations of what is unique about each of its data sets and a distance score that measures the amount of distribution shift between any two of its data sets. And MetaShift can be used to evaluate the models' performance across distribution shifts.

# Access
https://github.com/Weixin-Liang/MetaShift.git

## Data

## License
