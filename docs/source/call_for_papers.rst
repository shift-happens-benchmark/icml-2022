Call for Submissions
 ====================

.. note::

  We offer up to 5 free registrations to the ICML as prizes for outstanding submissions!
  
  We extended the submission deadlines:
  
  The new deadline for registering abstracts is **May 27**.
  All submissions (either an extended abstract, or a full submission in form of a technical report proof-of-concept implementation)
  are due on **June 3**.
  For authors of the `Neurips Datasets & Benchmarks Track submissions <https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks#:~:text=Abstract%20submission%20deadline%3A%20Monday%2C%20June,2022%2001%3A00%20PM%20PDT.>`__, we offer another deadline extension until **June 9**.
  
  Submit your contribution `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`_.

TL;DR
^^^^^

We accept the following submission types:
Full submissions and extended abstracts (`example <https://drive.google.com/file/d/1bRp0Pp2ek_KbuQILyNPuOgJcUD3EuCR3/view?usp=sharing>`__).

**Full submissions** consist of:

- a short technical report of the task, metrics and datasets
- a proof-of-concept implementation of the task, metrics and/or datasets, which we will later integrate into the ``shifthappens`` package.

Both components will be submitted via OpenReview, and reviewing is double-blind.

Please not that at submission time, a proof-of-concept implementation is sufficient.
Until the camera-ready deadline, the implementation should be comprised of a plugin to the ``shifthappens`` benchmark package, and the final version of the dataset needs to be provided.

**Extended abstracts** consist of:

- a one or two page PDF with one figure (`example <https://drive.google.com/file/d/1bRp0Pp2ek_KbuQILyNPuOgJcUD3EuCR3/view?usp=sharing>`__)

In an extended abstract, the authors describe an idea for a task, dataset or benchmark that will not be actively collected or integrated as part of the benchmark.

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
discussions about the ``shifthappens`` API, please feel free to write us as `shifthappens@bethgelab.org <mailto:shifthappens@bethgelab.org>`__
or `join our slack channel <https://join.slack.com/t/shifthappensicml2022/shared_invite/zt-16ewcukds-6jW6xC5DbtRvLCCkhZ~NLg>`__.

Deadlines
^^^^^^^^^^^^^^^^

You can find all deadlines as well as the submission page also directly `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__.

- `Abstract Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: May 27, 2022 (previously: May 17)
  
  - Register all short abstracts/intention to submit on OpenReview. We need this information to plan the review.  

- `Submission Deadline for all extended abstracts and full submissions <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: June 3, 2022 (previously: May 27)
- `Special submission Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__  for all authors of the `Neurips Datasets & Benchmarks Track submissions <https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks#:~:text=Abstract%20submission%20deadline%3A%20Monday%2C%20June,2022%2001%3A00%20PM%20PDT.>`__: June 9, 2022
- Reviews Posted: June 13, 2022
- Acceptance Notification: June 13, 2022
- Camera and Dataset Ready: July 1, 2022
- ICML 2022 Workshop dates: July 22

Please note that it is **not required** to post the final dataset by the submission deadline.
It is sufficient to start working on the final dataset collection as well as the finalizing the code associated with the submission after the acceptance notification until the camera ready deadline.

Please also note that it is **not required** to post a full implementation for adding your benchmark to the ``shifthappens`` package by the submission deadline. You can submit any implementation along with your submission, as long at is demonstrates the applicability of your model/task to the problem setup of the workshop. We will work with all authors towards adding all accepted submissions into the final benchmark package until the camera ready deadline.

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

Please refer to the `API Docs <https://shift-happens-benchmark.github.io/api.html>`__ for further information on how to implement benchmarks and datasets directly into the ``shifthappens`` package (not required, but encouraged at submission time).

