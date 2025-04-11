#%% Imports

import pandas as pd
from typing import List
from clusterer import Clusterer, ClusterType
from sklearn.linear_model import LogisticRegression

#%% Global Variables

class Model(Clusterer):
    """
    Model class that inherits from Clusterer.
    """

    def __init__(self,
        n_components: int | float = 0.8,
        cluster_model: ClusterType = ClusterType.kmeans,
        n_cluster: int = 3,
        random_state: int = 42,
        spot_dif: bool = True
        ):
        super().__init__(n_components=n_components,
                         cluster_model=cluster_model,
                         n_cluster=n_cluster,
                         random_state=random_state,
                         spot_dif=spot_dif)
        self.X_train_list = self.__get_list_by_cluster(self.X_train)
        self.X_test_list = self.__get_list_by_cluster(self.X_test)
        
    def __get_list_by_cluster(self, df : pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the dataframe into a list of dataframes by cluster.
        """
        return_list = []
        for i in range(self.n_cluster):
            df_temp = df[df["Cluster"] == i]
            df_temp = df_temp.drop(columns=["Cluster"])
            return_list.append(df_temp)
        return return_list
    
if __name__ == "__main__":
    model = Model(n_components=0.8,
                 cluster_model=ClusterType.kmeans,
                 n_cluster=3,
                 random_state=42,
                 spot_dif=False)