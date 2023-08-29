Installation
============

Installation was tested on Red Hat 7.6 with Python 3.8.5 and torch 1.4.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. CellMirror is implemented in the Pytorch framework. Please run CellMirror on CUDA if possible.

The ``CellMirror`` package can be installed using one of the following commands:

*************************
Installation via Anaconda
*************************

*********************
Installation via PyPi
*********************

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

    conda create -n CellMirror python=3.8.5 pip
    source activate
    conda activate CellMirror
    pip install -r used_package.txt

3. Install R packages

* Install tested on R = 4.0.0
* install.packages(c("Seurat", "ggplot2", "patchwork", "stringr", "magrittr", "here", "tidyverse"))