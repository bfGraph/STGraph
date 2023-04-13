![Seastar Banner](https://github.com/bfGraph/Seastar/blob/main/assets/Seastar%20Banner.png?raw=true)
[![Documentation Status](https://readthedocs.org/projects/seastar/badge/?version=latest)](https://seastar.readthedocs.io/en/latest/?badge=latest)

## How to contribute to Documentation

Tutorial for Python Docstrings can be found [here](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)

```
sphinx-apidoc -o docs/developers_guide/developer_manual/package_reference/ python/seastar/ -f
cd docs/
make clean
make html
```


## Attributions

Yidi WU. (2021). 21 April 2021Seastar: vertex-centric programming for graph neural networks. https://doi.org/10.1145/3447786.3456247
