Call for Submissions
===============

.. note::

    This page is still under construction and might contain inaccurate information.
    Please wait with the preparation of your submission until this note disappears.

Deadlines
^^^^^^^^^^^^^^^^

- Submission Deadline: June 1, 2022 (TODO)
- Acceptance notification: June 13, 2022 (TODO)
- ICML 2022 Workshop dates: July 22 and 23


Submission quickstart
^^^^^^^^^^^^^^^^^^^^^

Submissions consist of 

- A short report of the task, metrics and datasets,
- An implementation of the task, metrics and/or datasets, comprised of a plugin to the ``shifthappens`` benchmark package, and all required external data files.

Both components will be submitted via OpenReview, and reviewing is double-blind (**submission link tba**).
The workshop will not have any official proceedings, so it is non-archival.
Tasks that have been part of a recent submission/publication are allowed, in which case the overlapping works should be mentioned.

The implementation leverages our example API implementation:

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



Submission types (TODO when API is final)
^^^^^^^^^^^^^^^^

Besides compatibility to ImageNet scale models, the scope of possible
benchmarks and datasets is intentionally broad:

- Submissions that **provide their own evaluation criterion** and discuss its value in applications are particularly encouraged. Submissions should explain why the submitted dataset and metric are well-suited to inform about the specified property. Opening the benchmark to this form of submissions aims at reaching communities interested in problems besides standard, “accuracy-focused” settings.

- It is also encouraged to submit **datasets** that can be evaluated with one or more of the following **standard criteria**: This form of submission imposes a low bar on developing new code contributions and makes it possible to contribute in the form of well-recorded datasets.

- Third, in both cases, it is possible to re-submit existing and potentially published benchmarks, datasets, and evaluation settings, known only in a particular community and make these datasets available to a broader audience as part of a curated benchmark package. Examples include small datasets that test an interesting distribution shift, such as shifts occurring due to applications in the real world.

Within these three submission types, the design of the benchmark will
focus in particular on datasets falling into one or more of categories
below:

1. **Robustness to domain shifts (classification accuracy):** A labeled
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

Submissions are be allowed to contain multiple related datasets, e.g.,
a dataset like ImageNet-C could have been submitted as a collection of
15 evaluation datasets, corresponding to the different corruptions
ImageNet-C is comprised of.

Correspondingly, tasks to not need to output one single number. For example, a 
submission might include multiple (related) OOD datasets and demand that an
ideal model be not fooled by any of them. It might of course makes sense for a
multi-score benchmark to *also* calculate an average performance.

Code and Data Instructions (TODO when API is final)
^^^^^^^^^^^^^^^^

Submissions must include a link to the dataset (hosted on a suitable platform),
as well as code (building on top of the provided `reference implementation
<https://shift-happens-benchmark.github.io/icml-2022/>`__) for 
running the evaluation process.

Used data/images need to be usable for research purposes. Their license should
be stated in the report and README.

TODO: Code license

Report Instructions
^^^^^^^^^^^^^^^^


The short report should

- motivate why the submitted task is interesting,
- describe how the data was collected, as well as give an overview over the data,
- state how the data can be accessed,
- specify if there are special requirements on the models to be evaluated,
- detail the evaluation procedure,
- outline how the evaluation outputs can be interpreted,
- provide a short analysis how the task is challenging for some existing models
  (including the relevant provided ones),
- and establish context within related works.

The report should be limited to 2-4 pages without references.
If it includes an Appendix, it should be reserved for including additional 
sample images and technical details.

The report should be formatted according to the `ICML style instructions
<https://icml.cc/Conferences/2022/StyleAuthorInstructions>`__, by using the
provided `LaTeX files <https://media.icml.cc/Conferences/ICML2022/Styles/icml2022.zip>`__.



Evaluation Criteria
^^^^^^^^^^^^^^^^^^^


Submissions will be judged according to the following criteria:

1. **Correctness:** For labeled datasets, the labels should make sense to a
   human reviewer. For OOD datasets, no in-distribution objects can be
   visible on the images. During the review of large datasets, random
   samples and the worst mistakes of some models will be checked. The
   correctness will mainly be reviewed based on the submitted dataset
   and the technical report.

2. **Novelty**: Datasets which allow for a more insightful evaluation beyond
   the standard test accuracy of ImageNet are encouraged. 
   This can include well motivated new criteria, new datasets with emphasized 
   practical relevance, as well as tasks that demonstrate theoretically
   predicted weaknesses of certain popular models.
   
3. **Difficulty for current models**: If the task can easily be solved by
   humans but some models fail moderately or spectacularly, it is an
   interesting addition to the benchmark.
   This will be formally benchmarked by evaluating a set of standard models
   (including robustified, task specific ones) on the
   provided dataset. Together with the reference implementation,
   we have included
   (1) a set of (robustified) ResNet models,
   (2) models that provide an explicit OOD detection score, as well as
   (3) recent test-time adaptation methods.
   **Evaluation should be done by the authors and included in
   their technical report.**
   It should include all applicable reference models as well as relevant
   baselines and potentially proposed improvements.



Post-Submission and Reviewing
^^^^^^^^^^^^^^^^^^^

To eventually obtain a benchmark that is useful for and usable by the research
community, the review process will somewhat differ from usual paper review processes
*by centering around the code and data submission*.
Including the community in an open review process will be an opportunity
to increase chances for later adaptations of the benchmark. The tools
that will be made available for facilitating the reviewers' jobs will
later be released as open-source tools.

In more detail, reviewing will be done in the following stages:

1. As preparation for the review stage, all anonymized submissions will
   be public on OpenReview. In addition, the organizers will create (anonymized)
   pull requests on the benchmark repository based on the submissions.
   Authors are responsible for preparing their submissions accordingly,
   and documentation for doing this correctly (and testing the
   submission prior to uploading on OpenReview) will be made available
   on the workshop page.

2. In the reviewing phase, reviewers will judge the quality of both
   technical reports (on OpenReview) and submitted code (on GitHub),
   according to the above `evaluation criteria <call_for_papers.html#evaluation-criteria>`__.

3. In parallel to the reviewing phase, the workshop organizers will
   start running tests on the submitted benchmarks for an extended
   collection of established vision models.
   
4. While adding comments on OpenReview will be limited to the reviewers, code
   review (and proposal of improvements) on GitHub is open to the public
   — this also includes criticism of the data collection process
   described in the technical report. Thus, OpenReview comments are limited to
   a number of “formal” reviews. At the
   same time, public discussion — and community building relevant for the
   benchmark after the workshop ends — will be encouraged on GitHub.

5. In the discussion phase, authors are allowed to update both their
   technical report and the submitted code.

6. After the final decisions, all submissions will be de-anonymized both
   on OpenReview and on GitHub. 
   
7. The outlined review process will ensure
   that for this final set of camera-ready submissions, a set of
   datasets with reviewed descriptions (submitted reports), and
   high-quality code ready to merge into the benchmark will be
   available. After the camera-ready phase, and after ensuring technical
   soundness of the submitted pull requests, an initial version of
   the benchmark will be released that allows for contributing of additional
   models and techniques, as well as for making suggestions on improving
   the benchmarks and metrics.
   
8. Two weeks prior to the workshop, a *hackathon* aimed at
   community building around the benchmark will be hosted by the workshop
   organizers. For this, discussions will
   happen on GitHub, and the community will be able to contribute
   changes to the benchmark. The best contributions from this phase will
   get a short talk (time depends on the number of contributions) at the
   workshop.
   
9. All accepted submissions will be added to the shifthappens benchmark suite. 
   This will ensure that after the workshop all benchmarks are accessible
   to the community.

Removed Paragraphs (for now)
^^^^^^^^^^^^^^^^


We should note that we will make submission of code for review as easy
and convenient as possible for the authors: For example, the reference
package will make it possible to submit benchmark datasets with standard
metrics (e.g., accuracy on a new dataset), with a minimal code
submission, using helper functions already provided in the package.


Besides the robustness and out-of-distribution detection communities
directly addressed by the default benchmark items mentioned above, this
workshop pre-eminently is meant to bring together different communities
that can contribute assets in the form of datasets and interesting
evaluation tasks. For example, researchers who work primarily on
modeling 3D objects might provide an interesting puzzle piece to be
integrated in a comprehensive evaluation suite.

During the workshop, we will encourage discussion on (1) model
properties that are often overlooked when evaluating machine learning
models and should be included in a comprehensive benchmark, on (2)
important practical properties of evaluation datasets and criteria, and
on (3) currently unavailable evaluations that would be desirable to be
developed in the future. Furthermore, we will host an online forum in
the period between the camera-ready deadline and the workshop to
facilitate constructive discussions about the accepted datasets.

We ensure standardization of submitted datasets and evaluations
algorithms by providing a reference implementation with pre-defined
interfaces. These interfaces allow writing datasets and benchmarks that
are guaranteed to be compatible with a broad class of models. A critical
decision is to limit submissions to models compatible with ImageNet
pre-training: Given a batch of images, models will provide (at least)
class predictions and optionally features, class confidences, and an OOD
score. Given this information, each benchmark needs to define the
necessary mechanisms for evaluating and returning scores. Our reference
implementation (which will be extended in the coming weeks) is available
at https://github.com/shift-happens-benchmark/iclr-2022.
