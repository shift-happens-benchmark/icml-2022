.. Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet documentation master file, created by
   sphinx-quickstart on Fri Oct 29 17:59:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the API docs of the ShiftHappens benchmark!
======================================================

While the popularity of robustness benchmarks and new test datasets increased over the past years, 
the performance of computer vision models is still largely evaluated on ImageNet directly, or on simulated 
or isolated distribution shifts like in ImageNet-C. The goal of this two-stage workshop is twofold: First, 
we aim to enhance the landscape of robustness evaluation datasets for computer vision and devise new test settings
and metrics for quantifying desirable properties of computer vision models. Second, we expect that these improvements 
in the model evaluation lead to a better guided and, thus, more efficient phase for the development of new models. 
This incentivizes development of models and inference methods with meaningful improvements over existing approaches
with respect to a broad scope of desirable properties. Our goal is to bring the robustness, domain adaptation, and 
out-of-distribution detection communities together to work on a new broad-scale benchmark that tests diverse aspects 
of current computer vision models and guides the way towards the next generation of models.


Model implementations
---------------------

.. automodule:: shifthappens.models.base
   :members:
   :show-inheritance:
   :noindex:

.. automodule:: shifthappens.models.torchvision
   :members:
   :show-inheritance:
   :noindex:


Task implementations
--------------------

.. automodule:: shifthappens.tasks.base
   :members:
   :show-inheritance:
   :noindex:

