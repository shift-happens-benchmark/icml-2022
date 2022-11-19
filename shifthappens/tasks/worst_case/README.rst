Example for a Shift Happens task on ImageNet
==============================================
# Task Description
This task evaluates a set of metrics, mostly related to worst-class performance, as described in [1].
It is motivated by [2], where the authors note that using only accuracy as a metric is not enough to evaluate
 the performance of the classifier, as it must not be the same on all classes/groups.

## Evaluation Metrics
The evaluation metrics are "A", "WCA", "WCP", "WSupCA", "WSupCR",  "W10CR", "W100CR", "W2CA", "WCAat5", "W10CRat5", "W100CRat5", and their relevance is described in (J. Bitterwolf et al., "Classifiers Should Do Well Even on Their Worst Classes", https://openreview.net/forum?id=QxIXCVYJ2WP).

## Expected Insights/Relevance
To see the, how the model performs on its worst classes. The application examples are given in [1].


1. Classifiers Should Do Well Even on Their Worst Classes.
    J. Bitterwolf et al. 2022.

2. The Effects of Regularization and Data Augmentation are Class Dependent.
    R. Balestriero et al. 2022.

3. Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.
    C. Northcutt et al. 2021.

