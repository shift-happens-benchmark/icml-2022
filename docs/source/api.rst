.. Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet documentation master file, created by
   sphinx-quickstart on Fri Oct 29 17:59:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the API docs of the ShiftHappens benchmark!
======================================================

.. note::

    We are working on writing and extending the documentation for the API and on creating helper functions and classes to make the user experience more convenient.

**We aim to create a community-built benchmark suite for ImageNet models comprised of new datasets for OOD robustness
and detection, as well as new tasks for existing OOD datasets.**


While the popularity of robustness benchmarks and new test datasets
increased over the past years, the performance of computer vision models
is still largely evaluated on ImageNet directly, or on simulated or
isolated distribution shifts like in ImageNet-C. 

**Motivation:** The goal of this workshop is to enhance the landscape of robustness evaluation datasets for
computer vision and devise new test sets and metrics for quantifying desirable 
properties of computer vision models. Our goal is to bring the robustness, domain
adaptation, and out-of-distribution detection communities together to work on a new
**broad-scale benchmark** that tests diverse aspects of current computer
vision models and guides the way towards the next generation of models.


Model implementations
---------------------

.. automodule:: shifthappens.models.base
   :members:
   :show-inheritance:

.. automodule:: shifthappens.models.torchvision
   :members:
   :show-inheritance:


Task implementations
--------------------

.. automodule:: shifthappens.tasks.base
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

