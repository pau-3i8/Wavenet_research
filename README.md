# Wavenet_research

Code base for the paper: [Empirical modelling and prediction of neuronal dynamics]()
___

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Documentation](#documentation)
- [Authors](#authors)
___

## Description

The respository is a Wavelet Neural Network (Wavenet) used to predict biologically plausible input currents such as neuronal (single-cell) voltage traces. The neuron models studied are the Morris-Lecar, FitzHugh-Nagumo, FitzHugh-Nagumo-Rinzel and Wang model of a pyramidal neuron.

Since there are terascale models, out-of-core (ooc) computation has been used to compute the ordinary least squares for the models' training, using [Dask distributed](https://github.com/dask/distributed) in a single machine.

The repository provides scripts for training and evaluating the Wavenet in any similar neuron model with 2D or 3D ODEs.

The new desidered models should have the same structure as the ones presented in the Morris_Lecar.py, Nagumo_2D.py, Nagumo_3D.py and Wang_3D.py files. A configuration dictionary should be created following the preexisting ones in the configuration.py file.
___

## Installation

#### Prerequisites

The Python packages used are:
- bokeh >= 2.4.3
- cython >= 0.29.28
- dask == 2.30.0
- distributed == 2.30.0
- fastparquet >= 0.4.1
- graphviz >= 0.8.4
- llvmlite == 0.39.0
- matplotlib >= 3.3.2
- numba >= 0.56.0
- numcodecs >= 0.7.3
- numpy >= 1.22.0
- pandas >= 1.1.3
- psutil >= 5.9.0
- scipy >= 1.8.0
- terminaltables >= 3.1.0
- tqdm >= 4.50.2
- zarr >= 2.6.1

#### Plotting set-up

Installing latex for plotting
```
sudo apt-get install python3-graphviz python3-tk texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

#### Python >= 3.8

To avoid pickle related multiprocessing problems for parallel computing, it is advised to use Python >= 3.8. When updating from older Python versions remember to install llibpython3.* packages for Cython compilation.
```
sudo apt-get install libpython3-dev libpython3.8-dev
```

#### Compiling Cython .pyx code

Go to the directory /Wavenet and execute:
```
python3 setup.py build_ext --inplace
```
___

## Documentation

The code runs as is. Once compiled the Cython code in the [Wavenet](https://github.com/pau-3i8/Wavenet_research/tree/main/Wavenet) folder, to train and simulate a model it is necessary to execute a desidered model file. To train and simulate the Morris-Lecar model, for example:
```
python3 Morris_Lecar.py
```
In the configuration.py file there are all Wavenet and models' parameters to be changed with their description.
___

## Authors

#### Paper authors

- Pau Fisco - pau.fisco@upc.edu
- David Aquilué - daquilue99@gmail.com
- Néstor Roqueiro - nestor.roqueiro@ufsc.br
- Enric Fossas - enric.fossas@upc.edu
- Antoni Guillamon - antoni.guillamon@upc.edu

```
@misc{Wavenet research,
  title={Empirical modelling and prediction of neuronal dynamics},
  author={Fisco, Pau and
          Aquilu\'{e}, David and
          Roqueiro, N\'estor and
          Fossas, Enric and
          Guillamon, Antoni},
  journal={},
  year={2022}
}
```

#### Code author

- Pau Fisco - pau.fisco@upc.edu

```
@misc{Wavenet research,
  title={Wavenet research.},
  author={Fisco, Pau},
  journal={GitHub. Note: https://github.com/pau-3i8/Wavenet_research},
  volume={1},
  year={2022}
}
```

[Back To The Top](#Wavenet_research)
-->
