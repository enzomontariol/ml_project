# %% Imports


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import warnings

# %% Global Variables


warnings.filterwarnings("ignore")

data_path = "data/"
df_train_path = "df_imputed_train.csv"
df_test_path = "df_imputed_test.csv"
y_train_path = "y_train_binary.csv"
y_test_path = "y_test_binary.csv"

# %% Classes


@dataclass
class SampleType:
    """
    Class to hold sample type information.
    """

    train: str = "train"
    test: str = "test"


@dataclass
class ClusterType:
    """
    Class to hold cluster type information.
    """

    kmeans: str = "kmeans"
    gaussian_mixture: str = "gaussian_mixture"


class Clusterer:
    def __init__(
        self,
        n_components: int | float = 0.8,
        cluster_model: ClusterType = ClusterType.kmeans,
        n_cluster: int = 3,
        random_state: int = 42,
        spot_dif: bool = True,
    ):
        self.__random_state = random_state
        self.n_cluster = n_cluster
        self.__cluster_model = (
            KMeans(n_cluster, random_state=self.__random_state)
            if cluster_model == ClusterType.kmeans
            else GaussianMixture(
                n_components=n_cluster, random_state=self.__random_state
            )
        )
        self.__spot_dif = spot_dif
        self.__pca_main = PCA(n_components)
        self.X_train = self.__get_X(df_train_path, sample_type=SampleType.train)
        self.X_test = self.__get_X(df_test_path, sample_type=SampleType.test)
        self.__X_train_pca = self.__pca_main.transform(self.X_train)
        self.__X_test_pca = self.__pca_main.transform(self.X_test)
        self.__get_cluster_labels()

        self.y_train = self.__get_y(y_train_path, sample_type=SampleType.train)
        self.y_test = self.__get_y(y_test_path, sample_type=SampleType.test)

    def __get_X(self, sub_folder_path: str, sample_type: SampleType) -> pd.DataFrame:
        """
        Import imputed data from a CSV file.
        """
        df = pd.read_csv(data_path + sub_folder_path)
        df = self.__spot_diff_dropper(df)
        df["DELIVERY_START"] = pd.to_datetime(df["DELIVERY_START"], utc=True)
        df = df.set_index("DELIVERY_START")

        # Apply PCA transformation based on sample type
        if sample_type == SampleType.train:
            self.__pca_main.fit(df)

        return df

    def __spot_diff_dropper(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the 'spot_dif' column from the DataFrame.
        """
        if self.__spot_dif != True:
            for i in range(1, 19):
                self.df = self.df.drop(columns=f"spot_id_delta_lag_{i}", axis=1)
        return df

    def __get_y(self, sub_folder_path: str, sample_type: SampleType) -> pd.DataFrame:
        """
        Import target data from a CSV file.
        """
        df = pd.read_csv(data_path + sub_folder_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        
        if sample_type == SampleType.train:
            df = df[df.index.isin(self.X_train.index)]
        else:
            df = df[df.index.isin(self.X_test.index)]
        return df

    def __get_cluster_labels(self) -> None:
        """
        Get cluster labels for the given model and data.
        """
        self.__cluster_model.fit(self.__X_train_pca)
        self.X_train["Cluster"] = self.__cluster_model.predict(self.__X_train_pca)
        self.X_test["Cluster"] = self.__cluster_model.predict(self.__X_test_pca)

    def assert_cluster_model(
        self, model: ClusterType, k_range: range = range(2, 10)
    ) -> plt.Figure:
        """
        Assert the clustering model based on silhouette score, BIC and AIC.
        """
        return_list = self.__assert_cluster_number(model, k_range)
        silhouette_scores = return_list[0]
        if model == ClusterType.gaussian_mixture:
            bic_scores = return_list[1]
            aic_scores = return_list[2]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

            # Silhouette Score
            axes[0].plot(
                list(k_range), silhouette_scores, marker="o", label="Silhouette Score"
            )
            axes[0].set_title("Silhouette Score")
            axes[0].set_xlabel("Number of Clusters")
            axes[0].set_ylabel("Score")
            axes[0].grid(True)
            axes[0].legend()

            # BIC Score
            axes[1].plot(
                list(k_range), bic_scores, marker="o", color="orange", label="BIC Score"
            )
            axes[1].set_title("BIC Score")
            axes[1].set_xlabel("Number of Clusters")
            axes[1].set_ylabel("Score")
            axes[1].grid(True)
            axes[1].legend()

            # AIC Score
            axes[2].plot(
                list(k_range), aic_scores, marker="o", color="green", label="AIC Score"
            )
            axes[2].set_title("AIC Score")
            axes[2].set_xlabel("Number of Clusters")
            axes[2].set_ylabel("Score")
            axes[2].grid(True)
            axes[2].legend()

            plt.tight_layout()
            return fig
        else:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(
                list(k_range), silhouette_scores, marker="o", label="Silhouette Score"
            )
            plt.title("Silhouette Score")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Score")
            plt.grid(True)
            plt.legend()
            return fig

    def __assert_cluster_number(
        self, model_type: ClusterType, k_range: range = range(2, 10)
    ) -> None:
        """
        Assert the optimal number of clusters for our data based on BIC and silhouette scores.
        """
        silhouette_scores = []

        if model_type == ClusterType.gaussian_mixture:
            bic_scores = []
            aic_scores = []
            for k in k_range:
                model = GaussianMixture(
                    n_components=k, random_state=self.__random_state
                )
                model.fit(self.X_train)
                labels = model.predict(self.X_train)
                silhouette_scores.append(silhouette_score(self.X_train, labels))
                bic_scores.append(model.bic(self.X_train))
                aic_scores.append(model.aic(self.X_train))
            return_list = [silhouette_scores, bic_scores, aic_scores]
        else:
            for k in k_range:
                model = KMeans(n_clusters=k, random_state=self.__random_state)
                model.fit(self.X_train)
                labels = model.predict(self.X_train)
                silhouette_scores.append(silhouette_score(self.X_train, labels))
            return_list = [silhouette_scores]
        return return_list

