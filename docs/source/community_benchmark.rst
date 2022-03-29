Building the Community Benchmark 
==================


Post-Submission
^^^^^^^^^^^^^^^^^^^

- The `review process <call_for_reviewers.html#reviewing-process>`__ will center around the code and data submission. Including the community in an open review process will be an opportunity to increase chances for later adaptations of the benchmark. The tools that will be made available for facilitating the reviewers' jobs will later be released as open-source tools.

- In parallel to the reviewing phase, the workshop organizers will start running tests on the submitted benchmarks for an extended collection of established vision models.

- While adding comments on OpenReview will be limited to the reviewers, code review (and proposal of improvements) on GitHub is open to the public — this also includes discussion of the data collection process described in the technical report. 

- Public discussion — and community building relevant for the benchmark after the workshop ends — will be encouraged on GitHub.

- In the discussion phase, authors are allowed to update both their technical report and the submitted code.

- After the final decisions, all submissions will be de-anonymized both on OpenReview and on GitHub. 
   
- The review process will ensure that for this final set of camera-ready submissions, a set of datasets with reviewed descriptions (submitted reports), and high-quality code ready to merge into the benchmark will be available. After the camera-ready phase, and after ensuring technical soundness of the submitted pull requests, an initial version of the benchmark will be released that allows for contributing of additional models and techniques, as well as for making suggestions on improving the benchmarks and metrics.
   
- During a `community hackathon <community_benchmark.html#hackathon>`__, the benchmark will be polished.
   
- All accepted submissions will be part of the shifthappens benchmark suite. This will ensure that after the workshop all benchmarks are accessible to the community.

- $$$



Community Hackathon
^^^^^^^^^^^^^^^^

Two weeks prior to the workshop, a hackathon aimed at
community building around the benchmark will be hosted by the workshop
organizers. For this, discussions will
happen on GitHub, and the community will be able to contribute
changes to the benchmark. The best contributions from this phase will
get a short talk (time depends on the number of contributions) at the
workshop.
   
Ensuring Long-Term Access
^^^^^^^^^^^^^^^^


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
