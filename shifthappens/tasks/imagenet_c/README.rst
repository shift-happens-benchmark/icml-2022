Example for a Shift Happens task on ImageNet-C
==============================================
Evaluate the classification accuracy on a single corruption type of the ImageNet-C dataset [1]. Each corruption type has 5 different
severity levels. The raw images (before corruptions) in this dataset
come from the validation set of ImageNet.

While the dataset is ImageNet-C the task's definition is a bit different than the usual evaluation
paradigm:

- we allow the model to access the unlabeled test set separately for every corruption
- we allow it to make it's prediction based on a batch of samples coming from the same corruption type.

1. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
    Dan Hendrycks and Thomas Dietterich. 2019.

