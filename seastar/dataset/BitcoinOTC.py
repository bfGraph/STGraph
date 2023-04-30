class BitcoinOtcDataloader():
    def __init__(self):
        self._dataset = pd.read_csv('../../dataset/bitcoin/soc-sign-bitcoinotc.csv',header=0, names=['source', 'target', 'rating','time'])
        self._dataset.time = self._dataset.time / 10000000
        self._dataset.time = self._dataset.time.astype(int)

        lst = np.append(self._dataset.source.astype(int).unique(),self._dataset.target.astype(int).unique())
        lst = np.sort(np.unique(lst))
        
        id_dict = {}
        for i in range(len(lst)):
            id_dict[lst[i]] = i
        self.node_id_dict = id_dict
            
        self._dataset.source = self._dataset.source.apply(lambda x: self.node_id_dict[x])
        self._dataset.target = self._dataset.target.apply(lambda x: self.node_id_dict[x])
    
    def _get_edges(self):
        self._edges = []
        time_lst = self._dataset.time.unique()
        time_lst.sort()
        for time in time_lst:
            filtered_df = self._dataset.where(self._dataset.time == time).dropna()
            self._edges.append(
                np.array([filtered_df.source.astype(int).tolist(),filtered_df.target.astype(int).tolist()])
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in self._dataset.time.unique():
            filtered_df = self._dataset.where(self._dataset.time == time).dropna()
            self._edge_weights.append(
                np.array(filtered_df.rating.tolist())
            )

    def _get_targets_and_features(self):
        pass

    def get_dataset(self):
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return self._edges, self._edge_weights, None, None