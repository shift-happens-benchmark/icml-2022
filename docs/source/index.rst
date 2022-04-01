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

Participants will have the opportunity to co-author a paper summarizing their contributions to the benchmark suite.


Important Deadlines
-------------------

- Submission Deadline: mid-end May, 2022 (final date TBA)
- Reviews Posted: June 3, 2022
- Acceptance Notification: June 6, 2022
- Camera and Dataset Ready: July 1, 2022
- ICML 2022 Workshop Dates: July 22 and 23 (final date TBA)

Please note that it is **not required** to post the final dataset by the submission deadline since we are interested in new ideas for feasible datasets.
It is sufficient to start working on final dataset collections after the acceptance notification until the camera ready deadline.

Additional information about submission dates and the submission format can be found in :doc:`call_for_papers`.
Also, please consider our :doc:`call_for_reviewers`.


Prizes and Travel Grants
-------------------

To be announced soon.


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