#%% Imports

import pandas as pd
import numpy as np
from typing import List, Dict
from clusterer import Clusterer, ClusterType
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import all_estimators
from sklearn.metrics import balanced_accuracy_score


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
        self.classifier_models = {ExtraTreesClassifier(random_state=self.random_state):None,
                                 AdaBoostClassifier(random_state=self.random_state):None,
                                 LogisticRegression(random_state=self.random_state):None}
        
        self.train_models()        
        self.__sklearn_classifiers = self.__get_sklearn_classifiers()
        
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
    
    def train_models(self) -> None:
        """
        Trains the model using the ExtraTreesClassifier.
        """
        for i in self.classifier_models.keys():
            y_pred_test_list = []
            for j in range(self.n_cluster):
                X_train = self.X_train_list[j]
                y_train = self.y_train[self.y_train.index.isin(X_train.index)]
                
                i.fit(X_train, y_train)
                
                y_pred_test = i.predict(self.X_test_list[j])
                y_pred_test_list.append(y_pred_test)
                
            y_pred_test_all = np.concatenate(y_pred_test_list)
            score = balanced_accuracy_score(y_pred_test_all, self.y_test)
            self.classifier_models[i] = score
    
    def __get_sklearn_classifiers(self) -> Dict[str, object]:
        """
        Returns a list of sklearn classifiers.
        """
        estimators = all_estimators(type_filter='classifier')

        all_clfs = {}
        for name, ClassifierClass in estimators:
            try:
                clf = ClassifierClass()
                all_clfs[name] = clf
            except:
                continue
        return all_clfs
    
    def assert_best_classifier(self) -> pd.DataFrame:
        """
        Asserts the best classifier by balanced accuracy score.
        """
        output = pd.DataFrame(columns=["Classifier", "Score"])
        for classifier_name, classifier in self.__sklearn_classifiers.items():
            try:
                y_pred_test_list = []
                if hasattr(classifier, "random_state"):
                    classifier.set_params(random_state=self.random_state)
                for c in range(self.n_cluster):
                    X_train = self.X_train_list[c]
                    X_test = self.X_test_list[c]
                    y_train = self.y_train[self.y_train.index.isin(X_train.index)]
                    
                    classifier.fit(X_train, y_train)
                    
                    y_pred_test = classifier.predict(X_test)
                    y_pred_test_list.append(y_pred_test)

                y_pred_test_all = np.concatenate(y_pred_test_list)

                score = balanced_accuracy_score(y_pred_test_all, self.y_test)
                
                output = pd.concat([output, pd.DataFrame({"Classifier": [classifier_name], "Score": [score]})], ignore_index=True)
                
            except:
                continue
            
        return output.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
if __name__ == "__main__":
    model = Model(n_components=0.8,
                 cluster_model=ClusterType.kmeans,
                 n_cluster=3,
                 random_state=42,
                 spot_dif=False)
    print(model.assert_best_classifier())