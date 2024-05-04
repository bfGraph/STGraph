# STGraph Script Testing

Within this directory are scripts intended for manual execution by users or developers outside of the automated testing suite for STGraph. These scripts are primarily designed to assess the functionality of the following modules, whose correctness cannot be directly unit tested using PyTest. A table containing such modules can be found in the [List of Modules Tested](#list-of-modules-tested) section.

Additional scripts may be added as the project evolves.

## Usage

To execute the script tests, utilize the following command:

```
python3 stgraph_script.py [-h] [-v | --version VERSION] [-t | --testpack-names TESTPACK_NAMES]
```

For instance, to evaluate the GCN Dataloaders for STGraph version v1.1.0, execute:

```
python3 stgraph_script.py -v 1_1_0 -t gcn_dataloaders
```

Please ensure that the exact version of STGraph is installed within your virtual environment prior to conducting these tests.

### Command Line Arguments

| Argument             | Description                                            | Possible Values                                |
| -------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| -h, --help           | Obtain a brief description of the command              | -                                              |
| -v, --version        | Specify the version of the STGraph testpack to execute | `1_1_0`                                        |
| -t, --testpack-names | Provide a list of testpack names                       | `gcn_dataloaders`, `temporal_tgcn_dataloaders` |

## A Note to the Developers

The rest of the document outlines essential procedures that developers, contributors, and maintainers must follow for the STGraph project. As the project progresses, it is imperative to keep the testing script updated with new functionalities to prevent any potential issues for end-users.

### Post-Release Protocol

Following the release of a new version of STGraph, code owners are tasked with maintaining the integrity of the testpack folders corresponding to each version. These folders are located within the directory `tests/scripts`.

For instance, upon the release of STGraph version `v1.1.0`, and with the subsequent planning of version `v1.2.0`, the following steps are to be taken:

1. **Creation of New Test Pack:** A copy of the current version's testpack folder `v1_1_0` should be created and renamed to reflect the upcoming version `v1_2_0`.
2. **Development Phase Updates:** Any further enhancements or additions to the test scripts during the development phase of `v1.2.0` must be implemented within the designated folder. 

```
tests/
└── scripts
    ├── v1_1_0
    └── v1_2_0
```

By adhering to this protocol, the project maintains a structured and reliable testing framework, ensuring correctness and stability across successive releases.

### Test Script Creation Protocol

When preparing to write test scripts for newly implemented functionalities, ensure adherence to the following protocol

1. **Choose a Descriptive Name:** Select a meaningful and self-documenting name for your testpack, following the lower snake-case convention.
2. **Create Testpack Folder**: Within the current development version of STGraph, create a folder bearing the chosen testpack name.
3. **Script Creation**: Craft a Python script named `<testpack_name>.py`, containing the necessary testing logic.
4. **Supplementary Files**: Include any additional folders and files deemed necessary for testing within the testpack folder.
5. **Status Reporting**: Ensure that the Python script incorporates a mechanism to display the status of test cases effectively on the screen.
6. **Integration with STGraph Script**: Incorporate your newly created testpack as a selectable option for the `--testpack-names` command-line argument within `stgraph_script.py`

To get a better idea on how to develop your scripts, it is advised to refer to previously maintained scripts. This practice ensures uniformity and easier maintenance across all scripts.

## List of Modules Tested

| Module                   | Test Pack Name            | Initial Version Release |
| ------------------------ | ------------------------- | ----------------------- |
| GraphConv                | gcn_dataloaders           | `v1.1.0`                |
| CoraDataLoader           | gcn_dataloaders           | `v1.1.0`                |
| TGCN                     | temporal_tgcn_dataloaders | `v1.1.0`                |
| HungaryCPDataLoader      | temporal_tgcn_dataloaders | `v1.1.0`                |
| METRLADataLoader         | temporal_tgcn_dataloaders | `v1.1.0`                |
| MontevideoBusDataLoader  | temporal_tgcn_dataloaders | `v1.1.0`                |
| PedalMeDataLoader        | temporal_tgcn_dataloaders | `v1.1.0`                |
| WikiMathDataLoader       | temporal_tgcn_dataloaders | `v1.1.0`                |
| WindmillOutputDataLoader | temporal_tgcn_dataloaders | `v1.1.0`                |