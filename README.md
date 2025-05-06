# A Practical Implementation for Field-Based Computation of Core Loss in Permanent Magnet Synchronous Machines

This repository contains the code accompanying the case study of the [paper](https://doi.org/) (we will update the link when available).

## Code outline

The code is structured as follows:
+ [fea-data.h5](./fea-data.h5): Contains the stator and rotor core $B$-field data.
+ [num-example.py](./num-example.py): Contains the code that implements the paper case study.

## Running the code

The requirements for the case study are:
+ [numpy](https://numpy.org/doc/stable/index.html)
+ [h5py](https://www.h5py.org/)


## Reproducing the case study

Run the script to reproduce the paper case study:
+ [num-example.py](./num-example.py)

## Citation

```
@inproceedings{Beltran2025emloss,
  title={Practical Implementation for Field-Based Computation of Core Loss in Permanent Magnet Synchronous Machines}, 
  author={Beltrán-Pulido, Andrés and Aliprantis, Dionysios and Bilionis, Ilias and Chase, Nicholas and Munoz, Alfredo},
  year={2025,
  booktitle={2025 International Electric Machines and Drives Conference (IEMDC)}, 
  pages={1-6}
}
```