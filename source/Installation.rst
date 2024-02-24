Installation
============

Installation was tested on Red Hat 7.6 with Python 3.8.5 and torch 1.4.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. CellMirror is implemented in the Pytorch framework. Please run CellMirror on CUDA if possible.

The ``CellMirror`` package can be installed using the following commands:

************************
Installation from Wheels
************************
.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.

1. Grab source code of CellMirror

.. code-block:: bash
    :linenos:

    git clone https://github.com/JunjieXia14/CellMirror.git
    cd CellMirror

2. Install CellMirror in the virtual environment by conda

.. code-block:: bash
    :linenos:

    conda config --set channel_priority strict
    conda env create -f requirements.yml
    conda activate CellMirror

Other software dependencies are listed in "used_package.txt".

3. Install R packages

Our MNN program is deployed on R software and rpy2 library, please install r-base and related package dependecies via conda.

Run the following commands in Linux Bash Shell:

.. code-block:: bash
    :linenos:

    conda install r-base
    conda install r-dplyr (here, magrittr, tidyverse, batchelor, BiocParallel, FNN)

Or you can install these package dependencies by install.packages() and BiocManager::install() commands in R script.

.. tip::
    To reduce your waiting time, we recommend using the rpy2 library to call the path of R software installed in your existing virtual environment.