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

The repository provides scripts for training and evaluating the Wavenet with any similar neuron model with 2D or 3D ODEs.

The new desidered models should have the same structure as the ones presented in the [Morris_Lecar.py](https://github.com/pau-3i8/Wavenet_research/blob/master/Morris_Lecar.py), [Nagumo_2D.py](https://github.com/pau-3i8/Wavenet_research/blob/master/Nagumo_2D.py), [Nagumo_3D.py](https://github.com/pau-3i8/Wavenet_research/blob/master/Nagumo_3D.py) and [Wang_3D.py](https://github.com/pau-3i8/Wavenet_research/blob/master/Wang_3D.py) files. A configuration dictionary should be created following the preexisting ones in the [configuration.py](https://github.com/pau-3i8/Wavenet_research/blob/master/configuration.py) file.
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

Once downloaded the repository, go to the subdirectory /Wavenet and execute:
```
python3 setup.py build_ext --inplace
```
___

## Documentation

The code runs as is. Once compiled the [Cython code](https://github.com/pau-3i8/Wavenet_research/blob/master/Wavenet/activation_functions.pyx), to train and simulate a model it has to be executed the desidered model file. To train and simulate the Morris-Lecar model, for example:
```
python3 Morris_Lecar.py
```
In the [configuration.py](https://github.com/pau-3i8/Wavenet_research/blob/master/configuration.py) file there are all Wavenet and models' parameters to be changed with their description.
___

## Authors

#### Paper authors

- Pau Fisco - pau.fisco@upc.edu
- David Aquilué - daquilue99@gmail.com
- Néstor Roqueiro - nestor.roqueiro@ufsc.br
- Enric Fossas - enric.fossas@upc.edu
- Antoni Guillamon - antoni.guillamon@upc.edu

```
@unpublished{Wavenet_2023,
  title={Empirical modelling and prediction of neuronal dynamics},
  author={Fisco, Pau and
          Aquilu\'{e}, David and
          Roqueiro, N\'estor and
          Fossas, Enric and
          Guillamon, Antoni},
  year={2023},
  month={January}
}
```

#### Code author

- Pau Fisco - pau.fisco@upc.edu

```
@misc{Wavenet_code,
  title = {Wavenet research},
  author = {Fisco, Pau},
  howpublished = {GitHub},
  note = {https://github.com/pau-3i8/Wavenet\_research},
  volume = {1},
  year = {2022}
}
```

[Back To The Top](#Wavenet_research)
