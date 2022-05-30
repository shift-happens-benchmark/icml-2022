Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet
======================================================================

*ICML 2022 workshop*

**We aim to create a community-built benchmark suite for ImageNet models comprised of new datasets for OOD robustness
and detection, as well as new tasks for existing OOD datasets.**


While the popularity of robustness benchmarks and new test datasets
increased over the past years, the performance of computer vision models
is still largely evaluated on ImageNet directly, or on simulated or
isolated distribution shifts like in ImageNet-C. 

**Goal:** This workshop aims to enhance and consolidate the landscape of robustness evaluation datasets for
computer vision and collect new test sets and metrics for quantifying desirable or problematic
properties of computer vision models. Our goal is to bring the robustness, domain
adaptation, and out-of-distribution detection communities together to work on a new
**broad-scale benchmark** that tests diverse aspects of current computer
vision models and guides the way towards the next generation of models.

+-----------------------------------------------------------------------+
| |overview.svg|                                                        |
+=======================================================================+
| *Overview over the benchmark suite: You can contribute tasks and      |
| corresponding datasets highlighting                                   |
| interesting aspects of ImageNet-scale models. We will evaluate        |
| current and future models on the benchmark suite, testing their       |
| robustness, calibration, odd detection, and consistency, and make the |
| results intuitively accessible in the form of scorecards.*            |
+-----------------------------------------------------------------------+

**All accepted submissions will be part of the open-source** ``shifthappens`` **benchmark suite. This will ensure that after the workshop all benchmarks are accessible to the community.**

A central part of our package is to facilitate the evaluation of models on different datasets testing their generalization capabilities and providing fine-grained information on model performance using score-cards. To make sure all contributing authors as well as all authors of used (modified or not) pre-existing datasets will get credit for their efforts, we will release a bibtex file and a 'cite' macro for LaTeX which will include all contributions and underlying works.

In addition, participants will have the opportunity to co-author a paper summarizing the benchmark suite and all included contributions.

Focus Topics
-------------

Submissions to the benchmark suite will focus on datasets and evaluation algorithms falling into one or more of the categories
below:

1. **Robustness to domain shifts:** A labeled
   dataset where the labels are (a subset of) the 1000 labels of
   ImageNet-2012. Optionally, model calibration, uncertainty, or open
   set adaptation can be tested. We especially encourage submissions
   focusing on practically relevant distribution shifts.

2. **Out-of-distribution detection:** A labeled or unlabeled dataset of
   images that do not contain objects from any of the 1000 ImageNet-2012
   classes.

3. **New robustness datasets:** Beyond the standard robustness evaluation
   settings (with covariate shift, label shift, …), the workshop format
   enables submission of datasets that evaluate non-standard metrics
   such as the consistency of predictions, influence of spurious
   correlations in the dataset.

4. **New model characteristics:** Metrics and evaluation techniques that
   help examine the strengths, weaknesses and peculiarities of models in newly
   highlighted respects. Evaluations can utilize established datasets (or 
   subsets thereof) or come with their own dataset.
   
   
Submissions
-----------

The benchmark suite will be available on 
`GitHub <https://github.com/shift-happens-benchmark/icml-2022>`__.
The documentation for the benchmark's API is available `here <https://shift-happens-benchmark.github.io/icml-2022/>`__.
Please see our :doc:`call_for_papers` for more details.


For general questions about preparations of submissions, clarifications around the submission score and 
discussions about the ``shifthappens`` API, please feel free to write us as `shifthappens@bethgelab.org <mailto:shifthappens@bethgelab.org>`__
or `join our slack channel <https://join.slack.com/t/shifthappensicml2022/shared_invite/zt-16ewcukds-6jW6xC5DbtRvLCCkhZ~NLg>`__.

Important Deadlines
-------------------

You can find all deadlines as well as the submission page also directly `on OpenReview <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__.

- `Abstract Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: June 3, 2022 (previously: May 27)
- `Submission Deadline for all extended abstracts and full submissions <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__: June 3, 2022 (previously: May 27)
- `Special submission Deadline <https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens>`__  for all authors of the `Neurips Datasets & Benchmarks Track submissions <https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks#:~:text=Abstract%20submission%20deadline%3A%20Monday%2C%20June,2022%2001%3A00%20PM%20PDT.>`__: June 9, 2022
- Reviews Posted: June 13, 2022
- Acceptance Notification: June 13, 2022
- Camera and Dataset Ready: July 1, 2022
- ICML 2022 Workshop dates: July 22

Please note that it is **not required** to post the final dataset by the submission deadline since we are interested in new ideas for feasible datasets.
It is sufficient to start working on final dataset collections after the acceptance notification until the camera ready deadline.

Additional information about submission dates and the submission format can be found in :doc:`call_for_papers`.
Also, please consider our :doc:`call_for_reviewers`.


Prizes and Travel Grants
------------------------

We can offer up to 5 free registrations to the ICML for outstanding submissions.


Invited Speakers
----------------
.. raw:: html
    :file: speakers.html

Organizers
----------

.. raw:: html
    :file: organizers.html


.. |overview.svg| image:: overview.svg


.. toctree::
    :maxdepth: 2
    :hidden:

    Call for Submissions <call_for_papers>
    Call for Reviewers <call_for_reviewers>
    Benchmark API Docs <api>

.. meta::
      :title: Shift happens 2022: Crowdsourcing metrics and test datasets beyond ImageNet

      :description lang=en:
         This workshop at ICML 2022 aims to enhance and consolidate the landscape of robustness evaluation datasets for
         computer vision and collect new test sets and metrics for quantifying desirable or problematic
         properties of computer vision models.
