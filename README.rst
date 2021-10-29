Shift Happens: Crowdsourcing metrics and test datasets beyond ImageNet
======================================================================

Welcome! This is helper code for the ICLR 2022 workshop proposal on crowdsourcing 
novel metrics and test datasets for model evaluation on ImageNet scale.

Over the course of the workshop, the package will be populated with new tasks, metrics
and datasets that are suitable for understanding the properties of ImageNet scaled
computer vision models beyond the metrics that are typically reported.

Quickstart
----------

Install the package with

.. code::
    
    $ pip install shifthappens

Requirements (right now) are only `numpy`. Additional requirements will depend on the
tasks and models added to the benchmark in the future.

Adding models
-------------

Models should be added to the ``shifthappens.models`` module. An example implementation
for wrapping a torchvision model is given in ``shifthappens.models.torchvision``. Note
that implementations are framework agnostic; further options include tensorflow, and jax
models, for which we will add example implementations soon.

Adding Tasks
------------

Tasks should be added to the ``shifthappens.tasks`` module.

License
-------

All code in this repository is released under an Apache 2.0 license, which includes
external contributions. All data used for creating new benchmarks should minimally be
available without constrains for research purposes, and optionally free for commercial 
use as well.

Datasets `without an explicit license statement <https://choosealicense.com/no-permission/>`_ will not be accepted into the benchmark.
