from opencood.data_utils.datasets.basedataset.v2x_basedataset import V2XBaseDataset

class TestV2XDataset(V2XBaseDataset):
    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        
    def __getitem__(self, idx):
        # Get base data dictionary
        base_data_dict = self.retrieve_base_data(idx)
        
        # For testing, just return the raw data
        return base_data_dict