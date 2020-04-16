Installing razorback
====================

The stable version can be installed with:


.. code-block:: bash

  $ pip install razorback



Prerequisites
-------------

Razorback depends on:

  - `Python`_
  - `Numpy`_
  - `Scipy`_
  - `Matplotlib`_ (for visualization only)

.. One easy way to fulfil these dependances is using the `anaconda`_ distribution.


These packages are parts of the `Scipy stack`_, one easy way to install them is using the `anaconda`_ distribution.

.. Razorback works on all major platforms (linux, OSX, Windows)




Development
-----------

If you want to work on the development version, you can do:

.. code-block:: bash

  $ git clone https://github.com/BRGM/razorback.git
  $ pip install --upgrade --editable ./razorback


Also, you may need additional dependances:

  - to run the test suite:

    * `pytest`_

  - to build the documentation:

    * `Sphinx`_


.. _Python: http://www.python.org
.. _Numpy: http://www.numpy.org
.. _Scipy: https://www.scipy.org
.. _Matplotlib: https://matplotlib.org
.. _anaconda: https://www.continuum.io/downloads
.. _Scipy stack: https://www.scipy.org/install.html
.. _pytest: https://docs.pytest.org
.. _Sphinx: http://www.sphinx-doc.org
