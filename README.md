# On Optimal Regularization Parameters via Bilevel Learning
This python code can be used to recreate the numerical results of **On Optimal Regularization Parameters via Bilevel Learning** [1].

[1] Matthias J. Ehrhardt, Silvia Gazzola, Sebastian J. Scott. On Optimal Regularization Parameters via Bilevel Learning, arXiv preprint: [arXiv:2305.18394](https://arxiv.org/abs/2305.18394), 2023

## Installation
The main dependencies are
* Python [3.7.16]
* [Numpy](https://pypi.org/project/numpy/) [1.21.5]
* [Operator Discretization Library (ODL)](https://github.com/odlgroup/odl) [0.7.0]
* [Scipy](https://pypi.org/project/scipy/) [1.7.3]
* [Matplotlib](https://pypi.org/project/matplotlib/) [3.5.3]
* [TQDM](https://pypi.org/project/tqdm/) [4.64.1]
* [Pillow](https://pypi.org/project/Pillow/) [9.4.0]

and a full virtual environment can be created using the [environment.yml](environment.yml) file in conda by running the line
```
conda env create -f environment.yml
```
in the terminal.

## Getting started
The .py files called `RUNME_figure_X` can be used to generate numerics associated with Figure(s) X from the paper. For example, [RUNME_figure_4-5_7.py](RUNME_figure_4-5_7.py) can be used to generate relevant subfigures of Figure 4, Figure 5, and Figure 7. 
Due to the possibly long runtime to create some of the subfigures, one must uncomment specific lines of code at the start of these .py files to generate specific subfigures. For example, in the file [RUNME_figure_4-5_7.py](RUNME_figure_4-5_7.py) if the line 
```python
setup = {"ul":"MSE", "reg":"tikh", "A_choice":"id"}           # Fig 4a (1 minute)
```
is uncommented and the file is run, then Subfigure 4a will be created with an expected run time taking 30 seconds. 

## Citation and Acknowledgement

If you use the code for your work or if you found the code useful, please cite the following:

@article{ehrhardt2023optimal, title={On Optimal Regularization Parameters via Bilevel Learning}, author={Ehrhart, Matthias J. and Gazzola, Silvia and Scott, Sebastian J.}, journal={arXiv preprint arXiv:2305.18394}, year={2023} }