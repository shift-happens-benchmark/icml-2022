Call for Submissions
===============

.. note::

    This page is still under construction and might contain inaccurate information.
    Please wait with the preparation of your submission until this note disappears.

TL;DR
^^^^^^^^

Submissions consist of 

- A short report of the task, metrics and datasets,
- An implementation of the task, metrics and/or datasets, comprised of a plugin to the ``shifthappens`` benchmark package, and all required external data files.

Both components will be submitted via OpenReview, and reviewing is double-blind (**submission link TBA**).
The workshop will not have any official proceedings, so it is non-archival.
Tasks that have been part of a recent submission/publication are allowed, in which case the overlapping
works should be mentioned.

**Examples** of possible contributions:

- Collections of images to be evaluated w.r.t. one or more existing tasks like classification or OOD detection (e.g. ImageNet-A).
  Submissions of this type can consist of (labeled) images only.
- Re-definitions of tasks/new metrics on existing datasets
  (e.g. new calibration metrics, fairness metrics, ...).
- Completely new tasks and datasets that highlight differences between ImageNet models (see below for details).

Submissions creating links to communities interested in problems besides standard, “accuracy-focused” settings
are very welcome.

Deadlines
^^^^^^^^^^^^^^^^

- Submission Deadline: mid-end May, 2022 (final date TBA)
- Reviews Posted: June 3, 2022
- Acceptance Notification: June 6, 2022
- Camera and Dataset Ready: July 1, 2022
- ICML 2022 Workshop dates: July 22 and 23 (final date TBA)

Please note that it is **not required** to post the final dataset by the submission deadline.
It is sufficient to start working on the final dataset collection after the acceptance notification until the
camera ready deadline.


Detailed Information on Submission Types 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides compatibility to ImageNet scale models, the scope of possible
benchmarks and datasets is intentionally broad:

- Submissions that **provide their own evaluation criterion** and discuss its value in applications are particularly encouraged. Submissions should explain why the submitted dataset and metric are well-suited to inform about the specified property.

- It is also encouraged to submit **datasets** that can be evaluated with one of the following **standard criteria**:
 
  - Robustness to domain shifts (classification accuracy)
  - Out-of-distribution detection (AUROC, FPR, AUPR)

- Any other form of dataset and task that can be evaluated on a pre-trained (standard or non-standard training) ImageNet model.
There are *no constraints* on the possible metrics, as long as they are based on the features, class scores,
class uncertainties and in-distribution scores of such a model (also see our `reference implementation
<https://shift-happens-benchmark.github.io/icml-2022/>`__ for examples).

In all cases, it is possible to re-submit **existing and potentially published** benchmarks, datasets, and evaluation tasks to
consolidate them in one benchmark suite and/or if they are known only to a particular community. Examples include small datasets that test an
interesting distribution shift, such as shifts occurring due to applications in the real world, and
insightful benchmarks that you might have included in a publication highlighting the advantages or problems
of certain models.

Submissions are allowed to contain **multiple related datasets**, e.g.,
a dataset like ImageNet-C could have been submitted as a collection of
15 evaluation datasets, corresponding to the different corruptions
ImageNet-C is comprised of.

Correspondingly, tasks do not need to output one single number. For example, a 
submission might include multiple (related) OOD datasets and demand that an
ideal model be not fooled by any of them. It might of course makes sense for a
**multi-score** benchmark to *also* calculate an average performance.


Report Instructions
^^^^^^^^^^^^^^^^

The short report should

- motivate why the submitted task is interesting,
- describe how the data was collected, as well as give an overview over the data,
- state how the data can be accessed,
- specify if there are special requirements on the models to be evaluated,
- detail the evaluation procedure and outline how the evaluation outputs can be interpreted,
- provide a short analysis how the task is challenging for some existing models
  (including the relevant provided ones),
- and establish context within related works.

The report should be limited to 2-4 pages without references.
If it includes an Appendix, it should be reserved for additional 
sample images and technical details.

For the submission, the report should be formatted according to the `ICML style instructions
<https://icml.cc/Conferences/2022/StyleAuthorInstructions>`__, by using the
provided `LaTeX files <https://media.icml.cc/Conferences/ICML2022/Styles/icml2022.zip>`__.

Code and Data Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Submissions must include a link to the dataset (hosted on a suitable platform),
as well as code (building on top of the provided `reference implementation
<https://shift-happens-benchmark.github.io/icml-2022/>`__) for 
running the evaluation process. Datasets can be hosted on `zenodo <https://zenodo.org/>`__, 
`google drive <https://www.google.com/drive/>`__ (by only providing an anonymous google drive ID), or other platforms.

The data/images need to be usable for research purposes. Their license should
be stated in the report.

The implementation leverages our example `API <api.html>`__ implementation:



.. code:: python 

    from shifthappens.task import Result, Task, register
    
    @register
    class MyExampleTask(Task):

        def _evaluate(self, model):
            ...
            return Result(
                accuracy = 0.42,
                calibration = 0.44
            )


Evaluation Criteria for Submissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Submissions will be judged according to the following criteria:

1. **Correctness:** For labeled datasets, the labels should make sense to a
   human reviewer. For OOD datasets, no in-distribution objects can be
   visible on the images. During the review of large datasets, random
   samples and the worst mistakes of some models will be checked. The
   correctness will mainly be reviewed based on the submitted dataset
   and the technical report.

2. **Novelty**: Datasets which allow for a more insightful evaluation beyond
   the standard test accuracy of ImageNet are encouraged. 
   This can include well-motivated new criteria, new datasets with emphasized 
   practical relevance, as well as tasks that demonstrate theoretically
   predicted weaknesses of certain popular models.
   
3. **Difficulty for current models**: If the task can easily be solved by
   humans but some models fail moderately or spectacularly, it is an
   interesting addition to the benchmark.
   This will formally be benchmarked by evaluating a set of standard models
   (including robustified, task-specific ones) on the
   provided task.


