# STGraph Script Testing

Within this directory are scripts intended for manual execution by users or developers outside of the automated testing suite for STGraph. These scripts are primarily designed to assess the functionality of the following modules, whose correctness cannot be directly unit tested using PyTest:

1. Graph Neural Network (GNN) Layers
2. Temporal Graph Neural Network (TGNN) Layers
3. GNN Dataloaders
4. TGNN Dataloaders

Additional scripts may be added as the project evolves.

- [STGraph Script Testing](#stgraph-script-testing)
  - [Usage](#usage)
    - [Command Line Arguments](#command-line-arguments)


## Usage

To execute the script tests, utilize the following command:

```bash
python3 stgraph_script.py [-h] [-v | --version VERSION] [-t | --testpack-names TESTPACK_NAMES]
```

For instance, to evaluate the GCN Dataloaders for STGraph version v1.1.0, execute:

```bash
python3 stgraph_script.py -v 1_1_0 -t gcn_dataloaders
```

Please ensure that the exact version of STGraph is installed within your virtual environment prior to conducting these tests.

### Command Line Arguments

| Argument             | Description                                            | Possible Values                                |
| -------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| -h, --help           | Obtain a brief description of the command              | -                                              |
| -v, --version        | Specify the version of the STGraph testpack to execute | `1_1_0`                                        |
| -t, --testpack-names | Provide a list of testpack names                       | `gcn_dataloaders`, `temporal_tgcn_dataloaders` |