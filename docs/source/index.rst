Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet
======================================================================

*ICML 2022 workshop*

While the popularity of robustness benchmarks and new test datasets
increased over the past years, the performance of computer vision models
is still largely evaluated on ImageNet directly, or on simulated or
isolated distribution shifts like in ImageNet-C. The goal of this
two-stage workshop is twofold: First, we aim to enhance the landscape of
robustness evaluation datasets for computer vision and devise new test
settings and metrics for quantifying desirable properties of computer
vision models. Second, we expect that these improvements in the model
evaluation lead to a better guided and, thus, more efficient phase for
the development of new models. This incentivizes development of models
and inference methods with meaningful improvements over existing
approaches with respect to a broad scope of desirable properties. Our
goal is to bring the robustness, domain adaptation, and
out-of-distribution detection communities together to work on a new
broad-scale benchmark that tests diverse aspects of current computer
vision models and guides the way towards the next generation of models.

+-----------------------------------------------------------------------+
| |overview.svg|                                                        |autodoc_mock_imports
+=======================================================================+
| *Illustration of the envisioned procedure and outcome of this         |
| workshop: We will crowdsource and curate a collection of tasks and    |
| corresponding datasets highlighting interesting aspects of            |
| ImageNet-scale models. A set of reference models will be evaluated on |
| these datasets during the benchmark, yielding an initial set of       |
| scorecards for commonly used ImageNet models. Following the benchmark |
| creation, more models and new techniques can be evaluated, enabling a |
| more holistic view on the performance of practically relevant         |
| computer vision models.*                                              |
+-----------------------------------------------------------------------+

Submissions & Information
-------------------------

The benchmark will be available on
`github.com/shift-happens-benchmark/icml-2022 <https://github.com/shift-happens-benchmark/icml-2022>`__.
API docs are available on
`shift-happens-benchmark.github.io/icml-2022/ <https://shift-happens-benchmark.github.io/icml-2022/>`__.

The workshop aims to build up a range of evaluation datasets that
together allow for a detailed overview of a model’s strengths and
weaknesses across a variety of tasks. The workshop will result in a
software package of datasets and benchmarks interesting to a large
community dealing with ImageNet size models, including practitioners
interested in seeing practically relevant properties and trade-offs
between models.

Important Deadlines
-------------------

- Submission Deadline: TBD
- Acceptance notification: TBD
- ICML 2022 Workshop dates: July 22 and 23

Additional information about submission dates and the submission format can be found in :doc:`call_for_papers`.
Also, please consider our :doc:`call_for_reviewers`.

Invited Speakers
----------------

- Invited Talk 1: TBA
- Invited Talk 2: TBA
- Invited Talk 3: TBA
- Invited Talk 4: TBA


Organizers
----------

- Julian Bitterwolf
- Evgenia Rusak
- `Steffen Schneider <https://stes.io>`__
- `Roland S Zimmermann <https://rzimmermann.com/>`__
- `Matthias Bethge <http://bethgelab.org/>`__
- Wieland Brendel
- Matthias Hein 


.. |overview.svg| image:: overview.svg

.. toctree::
    :maxdepth: 2
    :hidden:

    Call for Papers <call_for_papers>
    Call for Reviewers <call_for_reviewers>
    Benchmark API Docs <api>