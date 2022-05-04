.. Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet documentation master file, created by
   sphinx-quickstart on Fri Oct 29 17:59:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the API docs of the ShiftHappens benchmark!
======================================================

We aim to create a community-built benchmark suite for ImageNet models comprised of new datasets for OOD robustness
and detection, as well as new tasks for existing OOD datasets.

While the popularity of robustness benchmarks and new test datasets
increased over the past years, the performance of computer vision models
is still largely evaluated on ImageNet directly, or on simulated or
isolated distribution shifts like in ImageNet-C. 

The goal of this workshop is to enhance the landscape of robustness evaluation datasets for
computer vision and devise new test sets and metrics for quantifying desirable 
properties of computer vision models. Our goal is to bring the robustness, domain
adaptation, and out-of-distribution detection communities together to work on a new
**broad-scale benchmark** that tests diverse aspects of current computer
vision models and guides the way towards the next generation of models.

Submissions to the workshop will be comprised of an addition of a :py:class:`Task <shifthappens.tasks.base.Task>`, which will be used to test the performance
of various computer vision models on a new evaluation task you specify with your submission. Below we provide documentation
for the ``shifthappens`` API.

Also make sure to look at the `examples <https://github.com/shift-happens-benchmark/icml-2022/tree/main/examples>`_
in the github repository. If in doubt or if the API is not yet sufficiently flexible to fit your needs, consider
opening an issue on github or join our slack channel.


Task implementations
--------------------

.. automodule:: shifthappens.tasks.base
   :members:
   :show-inheritance:
   :private-members:

.. automodule:: shifthappens.tasks.task_result
   :members:
   :show-inheritance:
   :private-members:

.. automodule:: shifthappens.tasks.metrics
   :members:
   :show-inheritance:
   :private-members:

Data loading
------------
.. automodule:: shifthappens.data.base
   :members:
   :show-inheritance:

.. automodule:: shifthappens.data.torch
   :members:
   :show-inheritance:

Model implementations
---------------------

.. automodule:: shifthappens.models.base
   :members:
   :show-inheritance:

.. automodule:: shifthappens.models.torchvision
   :members:
   :show-inheritance:


Benchmark
---------
.. automodule:: shifthappens.benchmark
   :members:
   :show-inheritance:

Storing tasks within the benchmark
----------------------------------
.. automodule:: shifthappens.task_data.task_metadata
   :members:
   :show-inheritance:

.. automodule:: shifthappens.task_data.task_registration
   :members:
   :show-inheritance: