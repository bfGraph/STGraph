# STGraph Documentation

Incase there are any major changes in the stgraph module or new modules are added, make sure to run the following beforehand

```shell
sphinx-apidoc -o docs/source stgraph
```

Then run the following to generate the docs

```shell
cd docs
make clean
make html
```