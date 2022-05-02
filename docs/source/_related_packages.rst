Related Software Packages
=========================

Below we list a set of related works in open-source software, benchmarks
and datasets released in the past years and gained popularity in
different communities. While some datasets are orthogonal to our effort,
we plan to seek active collaborations and discussions in case of
potential synergies. The organizing committee and invited speakers
already cover a considerable number of packages mentioned below.

`WILDS Benchmark <https://wilds.stanford.edu/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WILDS is “a benchmark of in-the-wild distribution shifts spanning
diverse data modalities and applications, from tumor identification to
wildlife monitoring to poverty mapping”. In contrast to the ShiftHappens
benchmark, WILDS is not primarily focused on the evaluation of
pre-trained ImageNet trained models but mainly considers the setting of
domain generalization on a broader range of tasks, which requires model
training.

However, we think that many synergies exist between our workshop goal
and the WILDS benchmark and are in contact with some of the authors who
will join the workshop as confirmed speakers.

`Robusta <https://github.com/bethgelab/robustness>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Robusta is a growing collection of helper functions, tools and methods
for robustness evaluation and adaptation of ImageNet scale models. The
focus is on simple methods that work at scale.

`Visual Decathlon <https://www.robots.ox.ac.uk/~vgg/decathlon/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Visual Decathlon challenge requires simultaneously solving ten image
classification problems representative of very different visual domains.
For this challenge, the participants were allowed to use the train and
validation splits of the different datasets to train their classifier
(or several classifiers). While the Visual Declathon requires a training
phase, the envisioned ShiftHappens benchmark focuses on evaluating
ImageNet pre-trained models.

`Visual Task Adaptation Benchmark (VTAB) <https://github.com/google-research/task_adaptation>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VTAB contains 19 challenging downstream tasks for evaluating vision
models. The tasks stem from different domains such as natural images,
artificial environments (structured), and images captured with
non-standard cameras (specialized). VTAB focuses on task adaptation,
needs a lot of compute for fine-tuning the models on the target tasks,
and is, therefore, orthogonal to our proposed benchmark (which will only
contain test datasets).

`ImageNet-C <https://github.com/hendrycks/robustness>`__, `ImageNet-P <https://github.com/hendrycks/robustness>`__, `ImageNet-R <https://github.com/hendrycks/imagenet-r>`__, `ImageNet-A <https://github.com/hendrycks/natural-adv-examples>`__, `ImageNet-O <https://github.com/hendrycks/natural-adv-examples>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ImageNet-C, -P, -R, -A, and -O are ImageNet-compatible datasets that are
highly relevant to the workshop, widely adopted in the community, and
will be included as reference datasets into the benchmark (all of them
are published under suitable open-source licenses). We invited Thomas G.
Dietterich, one of the ImageNet-C authors to share his thoughts about
his efforts in robustness evaluation during the workshop.

`ObjectNet <https://objectnet.dev/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to ImageNet-C and variants, ObjectNet is a currently isolated
benchmark dataset that fits the workshop’s scope. We will explore
possibilities for including ObjectNet in the reference implementation —
due to the special license of ObjectNet; this will require additional
attention from the ObjectNet authors.

`Model vs. Human <https://github.com/bethgelab/model-vs-human>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``modelvshuman`` package is centered around benchmarking the
similarity between ImageNet trained models and human subjects while
solving the same task. Co-organizers Wieland B. and Matthias B. are
actively involved in this project, and we are discussing possibilities
of leveraging synergies between this package and the ShiftHappens
benchmark.

`RobustBench <https://github.com/RobustBench/robustbench>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``robustbench`` package, initiated by co-organizer Matthias H.,
focuses on evaluating robustness against adversarial perturbations by
combining different state-of-the-art attack techniques. It also features
a leaderboard for robustness against the common corruptions in
CIFAR-C/Imagenet-C. As with other related packages, we will explore
synergies and potentially leverage functionality from the robustbench
package in our reference implementation.

`Foolbox <https://github.com/bethgelab/foolbox>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``foolbox`` package is a popular package around benchmarking
adversarial robustness. Co-organizer Wieland B. is one of the initiators
of this package. While we do not plan to focus specifically on
robustness to adversarial examples, we do not exclude the possibility
that some submissions make use of ``foolbox`` or related libraries for
robustness evaluation. However, an important criterion will be that
these datasets test practically relevant aspects (e.g., adding
adversarial patches to an image or other practically conceivable
scenarios).

`Timm <https://github.com/rwightman/pytorch-image-models>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``timm`` package is an increasingly popular package (>14,000 GitHub
stars) for state-of-the-art computer vision models trained with PyTorch.
The package includes reference results for robustness on
ImageNet-A,-R,-C [1]_, and we will explore possibilities of leveraging
the well-designed API and variety of models for our benchmark, e.g. by
making it easier to include ``timm`` models in the evaluation and
generation of model scorecards.

.. [1]
   See
   https://github.com/rwightman/pytorch-image-models/tree/master/results