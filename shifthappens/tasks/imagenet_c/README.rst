
Example for a Shift Happens task on ImageNet-C
==============================================

While the dataset is ImageNet-C the task's definition is a bit different than the usual evaluation
paradigm:

- we allow the model to access the unlabeled test set separately for every corruption
- we allow it to make it's prediction based on a batch of samples coming from the same corruption type.
