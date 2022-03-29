Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet
======================================================================

*ICML 2022 workshop*


While the popularity of robustness benchmarks and new test datasets
increased over the past years, the performance of computer vision models
is still largely evaluated on ImageNet directly, or on simulated or
isolated distribution shifts like in ImageNet-C. The goal of this
workshop is twofold: First, we aim to enhance the landscape of
robustness evaluation datasets for computer vision and devise new test
settings and metrics for quantifying desirable properties of computer
vision models. Second, we expect that these improvements in the model
evaluation will lead to a better guided and, thus, more efficient phase for
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

- Submission Deadline: June 1, 2022 (TODO)
- Acceptance notification: June 13, 2022 (TODO)
- ICML 2022 Workshop dates: July 22 and 23

Additional information about submission dates and the submission format can be found in :doc:`call_for_papers`.
Also, please consider our :doc:`call_for_reviewers`.

Invited Speakers
----------------

`Aleksander Mądry <https://people.csail.mit.edu/madry/>`__ [confirmed]
######################################################################
Aleksander Madry is the Cadence Design Systems Professor of Computing at MIT, leads the MIT Center for Deployable Machine Learning as well as is a faculty co-lead for the MIT AI Policy Forum. His research interests span algorithms, continuous optimization, and understanding machine learning from a robustness and deployability perspectives.

.. image:: organizer_speaker_pics/aleksander_madry.png
    :width: 150px
    :align: center

`Chelsea Finn <https://ai.stanford.edu/~cbfinn/>`__ [confirmed]
###############################################################
Chelsea Finn is an Assistant Professor in Computer Science and Electrical Engineering at Stanford University. Her lab, IRIS, studies intelligence through robotic interaction at scale, and is affiliated with SAIL and the ML Group. Chelsea also spends time at Google as a part of the Google Brain team. She is interested in the capability of robots and other agents to develop broadly intelligent behavior through learning and interaction.

.. image:: organizer_speaker_pics/chelsea_finn.jpg
    :width: 150px
    :align: center

`Ludwig Schmidt <https://people.csail.mit.edu/ludwigs/>`__ [confirmed]
######################################################################
Ludwig Schmidt is an assistant professor in the Paul G. Allen School of Computer Science & Engineering at the University of Washington. Ludwig’s research interests revolve around the empirical and theoretical foundations of machine learning, often with a focus on datasets, evaluation, and reliable methods. Ludwig completed his PhD at MIT under the supervision of Piotr Indyk and was a postdoc at UC Berkeley with Benjamin Recht and Moritz Hardt. Ludwig received a Google PhD fellowship, a Microsoft Simons fellowship, a new horizons award at EAAMO, a best paper award at ICML, and the Sprowls dissertation award from MIT.

.. image:: organizer_speaker_pics/ludwig_schmidt.jpg
    :width: 150px
    :align: center

`Alexei Efros <https://people.eecs.berkeley.edu/~efros/>`__ [TBD]
###################################################################################
..
	Alexei is a professor at EECS Department at UC Berkeley, where he is part of the Berkeley Artificial Intelligence Research Lab (BAIR). Before that, he spent nine years on the faculty of the Robotics Institute at CMU. Starting in 2007, Alexei have also been closely collaborating with Team WILLOW at École Normale Supérieure / INRIA in Paris. The central goal of Alexei’s research is to use vast amounts of unlabelled visual data to understand, model, and recreate the visual world around us. 

	.. image:: organizer_speaker_pics/alexey_efros.jpg
	    :width: 150px
	    :align: center


Organizers
----------

- `Julian Bitterwolf <https://scholar.google.com/citations?view_op=search_authors&mauthors=julian+bitterwolf&hl=en&oi=ao>`__
.. image:: organizer_speaker_pics/julian_bitterwolf.png
    :width: 150px
- `Evgenia Rusak <https://scholar.google.com/citations?user=XKc19kkAAAAJ&hl=en&oi=ao>`__
.. image:: organizer_speaker_pics/evgenia_rusak.png
    :width: 150px
- `Steffen Schneider <https://stes.io>`__
.. image:: organizer_speaker_pics/steffen_schneider.jpeg
    :width: 150px
- `Roland S Zimmermann <https://rzimmermann.com/>`__
.. image:: organizer_speaker_pics/roland_zimmermann.png
    :width: 150px
- `Matthias Bethge <http://bethgelab.org/>`__
.. image:: organizer_speaker_pics/matthias_bethge.png
    :width: 150px
- `Wieland Brendel <https://scholar.google.com/citations?user=v-JL-hsAAAAJ&hl=en&oi=ao>`__
.. image:: organizer_speaker_pics/wieland_brendel.png
    :width: 150px
- `Matthias Hein <https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/maschinelles-lernen/team/prof-dr-matthias-hein/>`__
.. image:: organizer_speaker_pics/matthias_hein.png
    :width: 150px

.. |overview.svg| image:: overview.svg


.. toctree::
    :maxdepth: 2
    :hidden:

    Call for Submissions <call_for_papers>
    Reviewing <call_for_reviewers>
    Community Benchmark <community_benchmark>
    Main Event <main_event>
    Benchmark API Docs <api>
