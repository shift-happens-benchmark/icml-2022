Call for Submissions
====================

.. note::

  We offer up to 5 free registrations to the ICML as prizes for outstanding submissions!
  
  We extended the submission deadlines:
  
  All submissions (either an extended abstract, or a full submission in form of a technical report proof-of-concept implementation)
  are due on **June 9** (previously: June 3).
  For authors of the `Neurips Datasets & Benchmarks Track submissions <https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks#:~:text=Abstract%20submission%20deadline%3A%20Monday%2C%20June,2022%2001%3A00%20PM%20PDT.>`__, we offer another deadline extension until **June 9**.
  
  Submit your contribution `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`_. In case of technical issues, please contact us at `shifthappens@bethgelab.org <mailto:shifthappens@bethgelab.org>`__.

TL;DR
^^^^^

We accept the following submission types:
Full submissions and extended abstracts (`example <https://drive.google.com/file/d/1bRp0Pp2ek_KbuQILyNPuOgJcUD3EuCR3/view?usp=sharing>`__).

**Full submissions** consist of:

- a short technical report of the task, metrics and datasets
- an implementation of the task, metrics and/or datasets, and all required external data files.

Both components will be submitted via OpenReview, and reviewing is double-blind.

Please note that at submission time, a proof-of-concept implementation is sufficient.
Until the camera-ready deadline, the implementation should be comprised of a plugin to the ``shifthappens`` benchmark package, and the final version of the dataset needs to be provided.

**Extended abstracts** consist of:

- a one or two page PDF with one figure (`example <https://drive.google.com/file/d/1bRp0Pp2ek_KbuQILyNPuOgJcUD3EuCR3/view?usp=sharing>`__)

In an extended abstract, the authors describe an idea for an interesting dataset or task they intend working on in the near future to get peer feedback.

**Submission**

Please submit your paper `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__. 
The workshop will not have any official proceedings (besides OpenReview), so it is non-archival.
Tasks that have been part of a recent submission or publication are allowed and encouraged.

**Examples** of possible contributions:

- Collections of images to be evaluated w.r.t. one or more existing tasks like classification or OOD detection (e.g. ImageNet-A).
  Submissions of this type can consist of (labeled) images only.
- Re-definitions of tasks/new metrics on existing datasets
  (e.g. new calibration metrics, fairness metrics, ...).
- Completely new tasks and datasets that highlight differences between ImageNet models (see below for details).

Submissions creating links to communities interested in problems besides standard, “accuracy-focused” settings are very welcome and encouraged.

For general questions about preparations of submissions, clarifications around the submission score and 
discussions about the ``shifthappens`` API, please feel free to write us at `shifthappens@bethgelab.org <mailto:shifthappens@bethgelab.org>`__
or `join our slack channel <https://join.slack.com/t/shifthappensicml2022/shared_invite/zt-16ewcukds-6jW6xC5DbtRvLCCkhZ~NLg>`__.

Deadlines
^^^^^^^^^^^^^^^^

You can find all deadlines as well as the submission page also directly `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__.

- `Abstract Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: June 9, 2022 (previously: June 3, May 27)
- `Submission Deadline for all extended abstracts and full submissions <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: June 9, 2022 (previously: June 3, May 27)
- `Special submission Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__  for all authors of the `Neurips Datasets & Benchmarks Track submissions <https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks#:~:text=Abstract%20submission%20deadline%3A%20Monday%2C%20June,2022%2001%3A00%20PM%20PDT.>`__: June 9, 2022
- Reviews Posted: June 13, 2022
- Acceptance Notification: June 13, 2022
- Camera and Dataset Ready: July 1, 2022
- ICML 2022 Workshop dates: July 22

Please note that it is **not required** to post the final dataset by the submission deadline.
It is sufficient to start working on the final dataset collection as well as the finalizing the code associated with the submission after the acceptance notification until the camera ready deadline.

Please also note that it is **not required** to post a full implementation for adding your benchmark to the ``shifthappens`` package by the submission deadline. You can submit any implementation along with your submission, as long at is demonstrates the applicability of your model/task to the problem setup of the workshop. We will work with all authors towards adding all accepted submissions into the final benchmark package until the camera ready deadline.


Detailed Information on Submission Types and Topics 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Submission Types
****************

Extended abstracts are one to two page descriptions of an idea for a task, dataset or benchmark that will not be actively collected and integrated as part of the benchmark. The submission should contain at least one summary figure and should use the standard ICML submission template.
You can find an example `here <https://drive.google.com/file/d/1bRp0Pp2ek_KbuQILyNPuOgJcUD3EuCR3/view?usp=sharing>`__.
If you are submitting an extended abstract, no additional supplementary material is allowed.

Full submissions are comprised of a 2-4 page technical report, along with a code contribution submitted as supplementary material. The code should be a proof-of-concept implementation, which we will later integrate into the ``shifthappens`` package. If you submit a dataset, provide at least sample data along with your code as supplementary material. For large submissions, we encourage to host data externally and include instructions for downloading.

Topics
******

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
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Submissions** should demonstrate the full capability of your dataset/task/benchmark, but do not need to contain a final implementation, a full dataset, etc. yet. Make sure to submit a code sample and (parts of) the dataset as supplementary material to your paper submission, directly on OpenReview. Please make sure that from the submitted code it becomes clear how a model would be evaluated. While we invite you to directly build on top of the provided `reference implementation <https://shift-happens-benchmark.github.io/icml-2022/>`__, this is not a requirement at submission time (for example, it it acceptable to provide code for a reference run of a ResNet50 model, or whatever is suitable for your task, even outside the ``shifthappens`` package). If you have questions about implementation, please do not hesitate to reach out via email or our slack channel. We will continue to assist authors of accepted submissions to make their submission ready for integration to the ``shifthappens`` package.

Until the camera-ready deadline, all submissions need to be updated to include a link to the dataset (hosted on a suitable platform),
as well as code (building on top of the provided `reference implementation
<https://shift-happens-benchmark.github.io/icml-2022/>`__) for 
running the evaluation process. Datasets can be hosted on `zenodo <https://zenodo.org/>`__, 
`google drive <https://www.google.com/drive/>`__ (by only providing an anonymous google drive ID), or other platforms.

The data/images need to be usable for research purposes. Their license should
be stated in the report.

Please refer to the `API Docs <https://shift-happens-benchmark.github.io/api.html>`__ for further information on how to implement benchmarks and datasets directly in the ``shifthappens`` package (not required, but encouraged at submission time).


Evaluation Criteria for Submissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
