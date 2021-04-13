Command line interface
======================

The :code:`rzb` command
-----------------------

Razorback provides some command line tools under the :code:`rzb` command.
Typing the command shows the help message with the list of available commands :

.. code-block:: bash

  $ rzb
  Usage: rzb [OPTIONS] COMMAND [ARGS]...
  
    The razorback command line interface.
  
  Options:
    -h, --help  Show this message and exit.
  
  Commands:
    path       Manipulate data path.
    version    Razorback installed version.

Getting the version number with :code:`rzb version`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the number of the installed version :

.. code-block:: bash

  $ rzb version
  razorback 0.4.0

Manipulating the data path with :code:`rzb path`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`rzb path` provides several subcommands for manipulating the data path.
Just typing :code:`rzb path` performs no action but shows the help and the list of available commands :

.. code-block:: bash

  $ rzb path
  Usage: rzb path [OPTIONS] COMMAND [ARGS]...
  
    Manipulate data path.
  
    Data path is either global or local. If the local path is not available,
    the global path is used instead.
  
    The path commands depend on the current directory where they are executed.
  
  Options:
    -c, --create  Create the local path if missing.
    -h, --help    Show this message and exit.
  
  Commands:
    base      Current base data path.
    metronix  Current path for Metronix calibration files.


Without options, :code:`rzb path base` and :code:`rzb path metronix` just show the path where data, like calibration files, will be searched. This path depends on the working directory.

The option :code:`--create` will create the corresponding local path.


Extending the command line interface
------------------------------------

You can add commands to the :code:`rzb` command line interface by using `Click <https://click.palletsprojects.com/>`_ and `setuptools entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`_.

Let us look at an example.
We have a simple `Click <https://click.palletsprojects.com/>`_ program in our package:

.. code-block:: python

  # mypkg/rzb_cli.py
  import click

  @click.command('say-hello')
  def cli():
      click.echo('Hello world !')

We also have a `setup.py` for installing our package.
To extend the :code:`rzb` command, we need to informs the `setup()` function in the following way:

.. code-block:: python

  # setup.py
  setup(
    # ...
    entry_points={
        'rzb.commands': [
            'say-hello=mypkg.rzb_cli:cli',
        ]
    },
  )

Once `mypkg` is installed (:code:`python setup.py install` or :code:`pip install .`), the :code:`rzb` command can now expose our new subcommand:

.. code-block:: bash

  $ rzb say-hello
  Hello world !
