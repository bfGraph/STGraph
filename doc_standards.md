# Standards and rules followed for documenting

## How to document classes

```
r"""Short info about the class.

    Descriptive info about the class

    Parameters
    ----------
    mode : str, optional
        Should be one of ['train', 'dev', 'test', 'tiny']
        Default: train
    glove_embed_file : str, optional
        The path to pretrained glove embedding file.
        Default: None
    vocab_file : str, optional
        Optional vocabulary file. If not given, the default vacabulary file is used.
        Default: None
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    vocab : OrderedDict
        Vocabulary of the dataset
    num_classes : int
        Number of classes for each node
    pretrained_emb: Tensor
        Pretrained glove embedding with respect the vocabulary.
    vocab_size : int
        The size of the vocabulary

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # get dataset
    >>> train_data = SSTDataset()
    >>> dev_data = SSTDataset(mode='dev')
    >>> test_data = SSTDataset(mode='test')
    >>> tiny_data = SSTDataset(mode='tiny')
    >>>
    >>> len(train_data)
    8544
    >>> train_data.num_classes
    5
    >>> glove_embed = train_data.pretrained_emb
    >>> train_data.vocab_size
    19536
    >>> train_data[0]
    Graph(num_nodes=71, num_edges=70,
      ndata_schemes={'x': Scheme(shape=(), dtype=torch.int64), 'y': Scheme(shape=(), dtype=torch.int64), 'mask': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={})
    >>> for tree in train_data:
    ...     input_ids = tree.ndata['x']
    ...     labels = tree.ndata['y']
    ...     mask = tree.ndata['mask']
    ...     # your code here
    """
```
