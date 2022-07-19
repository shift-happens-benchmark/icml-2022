Welcome to the Shift Happens contributing guide
===============================================

Thanks for you interest in extending the `shifthappens` benchmark. Please read through the following
guidelines for information on how to prepare your pull request.

Types of contributions:
    1. Tasks, benchmarks and datasets
    2. Enhancement, documentation, and bugfixes


General information
-------------------

All PR are tested against a list of checks via github actions. An easy way to check for 
obvious mistakes is by installing the `pre-commit <https://pre-commit.com/>`_ tool and
running

.. code:: 

    $ pre-commit

before you commit and push your code.

When submitting a PR, please enable the "Allow edits by maintainer" option.


Contributing tasks, benchmarks and datasets
-------------------------------------------

To submit a new task, benchmark, or dataset, please follow the following steps:

Create a new python package in `shifthappens/tasks/<your_task>`.
Use the following structure:

.. code::

    shifthappens/tasks/<your_task>
        __init__.py             # this file should contain or import your benchmark,
                                  as well as registering it to the evaluation suite.
        __main__.py               if you generate data for your task,
                                # provide a command line utility in this
                                  file.
        <util_1>.py             # add any number of additional utility files as
        <util_2>.py               required.
        ...
        README.{md,rst,...}     # provide a README file of any format
        CITATION.cff            # provide a citation file in bibtex format with a link
                                  to the paper people should cite.

You can add as many utility or helper scripts as necessary.

Registration of the task should happen in the ``__init__.py`` method of your package.
Place the evaluation/helper code in other modules as you see fit. Please make sure to add
a doc string to the top of your ``__init__.py`` file, which is used to auto-generate
the documentation.

Make sure to then import your package in the ``shifthappens/tasks/__init__.py`` file of
the benchmark. This ensures that your registration code is loaded whenever the 
``shifthappens.task`` package is imported.

.. code:: python 

    #file: shifthappens/tasks/__init__.py
    ...
    from shifthappens.tasks import your_task
    ...

If you generated a new dataset, code for reproducing the generation
should go into your directory as well. In this case, you should provide a command
line interface via the ``__main__.py`` file, which will later make your scripts callable
as ``python -m shifthappens.tasks.<your_task> --arg1 val ...``.

If users of the benchmark should cite your work in publications, please make sure to
provide a citation file in `citation file format (cff) <https://citation-file-format.github.io/>`_.

If you want to specify additional information on how to use your benchmark, please add a
README file of any file format.

We are currently looking into options for how to setup unit and integration tests and will setup this note accordingly. If you have suggestions, feel free to create a PR for adapting this contribution guide.

Finally, update the ``LICENSE`` file in the repository root, and also the ``README.rst`` file 
in the repository root with the name of contributors you would like to add. Please note that
we can only add contributions that are compatible with the Apache 2.0 license right now.
If you want to add code that cannot be (re-) licensed in this way, please add a note and we 
will find a solution in the PR discussion.


Contributing enhancements, documentation, and bugfixes
------------------------------------------------------

For all other contributions, you do not need to follow any particular workflow. Please
add your contribution, and open a PR. If you contribute code, please make sure to add tests
covering your code.

The PR description should briefly outline what feature you add/bug you fix, and a test plan,
if applicable.
