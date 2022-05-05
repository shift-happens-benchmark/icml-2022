Shift Happens: Crowdsourcing metrics and test datasets beyond ImageNet
======================================================================

Welcome! This is the official code for the ICML 2022 workshop on crowdsourcing 
novel metrics and test datasets for model evaluation on ImageNet scale.

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

Over the course of the workshop, the package will be populated with new tasks, metrics
and datasets that are suitable for understanding the properties of ImageNet scaled
computer vision models beyond the metrics that are typically reported.

How to use this benchmark
----------
For now, you can use the benchmark by installing it from this repository:

.. code::

    $ pip install git+https://github.com/shift-happens-benchmark/icml-2022.git

After the workshop you will be able to use this package and all its included tasks with:

.. code::
    
    $ pip install shifthappens


How to contribute
-----------------

Since the aim of this workshop & package is to build a unified platform for datasets
investigating and highlighting interesting properties of ImageNet-scale vision models,
we are looking for **your** contribution. If you decide to contribute a new task to the 
benchmark before ICML 2022 please consider officially submitting it to the workshop - for
more details see `here <https://shift-happens-benchmark.github.io/call_for_papers.html>`_.


Adding tasks and datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

Tasks in this benchmark package should highlight interesting properties of vision models.
For one, this means that you can integrate new datasets you built. In addition, you can also
propose new evaluation schemes (i.e. new tasks) for already existing datasets, like test-time adaptation evaluation
on robustness datasets. You can think about examples/scenarios that might be of interest for industrial
applications just as well as purely academic examples - as long as the new tasks/datasets highlight 
an interesting behavior of existing models, it fits into this package! 

New tasks should be added to the ``shifthappens.tasks`` module.

Please refer to the `API documentation <https://shift-happens-benchmark.github.io/api.html>`_ for 
more details, as well as minimal examples. Moreover, inside the `examples <examples>`_ folder you can 
find implementations example implementations of tasks for the benchmark.

Adding models
^^^^^^^^^^^^^

Models should be added to the ``shifthappens.models`` module. An example implementation
for wrapping a torchvision model is given in ``shifthappens.models.torchvision``. Note
that implementations are framework agnostic; further options include TensorFlow, and jax
models, for which we will add example implementations soon.

License
-------

All code in this repository is released under an Apache 2.0 license, which includes
external contributions. All data used for creating new benchmarks should minimally be
available without constrains for research purposes, and optionally free for commercial 
use as well.

Datasets `without an explicit license statement <https://choosealicense.com/no-permission/>`_ 
will not be accepted into the benchmark.
