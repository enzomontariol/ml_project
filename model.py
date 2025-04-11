# %% Imports

import pandas as pd
import numpy as np
from typing import List, Dict
from clusterer import Clusterer, ClusterType
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import all_estimators
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# %% Global Variables


class Model(Clusterer):
    """
    Model class that inherits from Clusterer.
    """

    def __init__(
        self,
        n_components: int | float = 0.8,
        cluster_model: ClusterType = ClusterType.kmeans,
        n_cluster: int = 3,
        random_state: int = 42,
        spot_dif: bool = True,
    ):
        super().__init__(
            n_components=n_components,
            cluster_model=cluster_model,
            n_cluster=n_cluster,
            random_state=random_state,
            spot_dif=spot_dif,
        )
        self.X_train_list = self.__get_list_by_cluster(self.X_train)
        self.X_test_list = self.__get_list_by_cluster(self.X_test)
        self.classifier_models = {
            ExtraTreesClassifier(random_state=self.random_state): None,
            AdaBoostClassifier(random_state=self.random_state): None,
            LogisticRegression(random_state=self.random_state): None,
        }
        self.train_classifier_models()
        self.__sklearn_classifiers = self.__get_sklearn_classifiers()
        
        self.__lstm_results = self.__get_lstm_model()
        self.lstm_models = self.__lstm_results[0]
        self.lstm_scores = self.__lstm_results[1]
        self.lstm_global_score = self.__get_lstm_global_score()

    def __get_list_by_cluster(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the dataframe into a list of dataframes by cluster.
        """
        return_list = []
        for i in range(self.n_cluster):
            df_temp = df[df["Cluster"] == i]
            df_temp = df_temp.drop(columns=["Cluster"])
            return_list.append(df_temp)
        return return_list

    def train_classifier_models(self) -> None:
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
        estimators = all_estimators(type_filter="classifier")

        all_clfs = {}
        for name, ClassifierClass in estimators:
            try:
                clf = ClassifierClass()
                all_clfs[name] = clf
            except:
                continue
        return all_clfs

    def __get_lstm_model(self):
        models = {}
        scores = {}
        for i in range(self.n_cluster):
            X_train = self.X_train_list[i]
            y_train = self.y_train[self.y_train.index.isin(X_train.index)]
            X_test = self.X_test_list[i]
            y_test = self.y_test[self.y_test.index.isin(X_test.index)]

            model, balanced_accuracy = (
                self.__create_and_train_lstm_model(X_train, y_train, X_test, y_test)
            )
            models[i] = model
            scores[i] = balanced_accuracy
            
        return models, scores
    
    def __get_lstm_global_score(self):
        """
        Returns the global score of the LSTM model.
        """
        predictions = []
        for i in range(self.n_cluster):
            X_test = self.X_test_list[i]
            X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Reshape X_test for LSTM input
            model = self.lstm_models[i]
            predictions.append(model.predict(X_test, verbose = 0).flatten())
        y_pred_all = np.concatenate(predictions)
        y_pred_binary = (y_pred_all > 0.5).astype(int)  # Convert probabilities to binary predictions
        lstm_global_score = balanced_accuracy_score(y_pred_binary, self.y_test)
        return lstm_global_score

    def __create_and_train_lstm_model(
        self, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster
    ):
        # Suppose que tu veux 1 seul timestep par séquence (chaque observation = 1 timestep)
        X_train_cluster = X_train_cluster.values.reshape((X_train_cluster.shape[0], 1, X_train_cluster.shape[1]))
        X_test_cluster = X_test_cluster.values.reshape((X_test_cluster.shape[0], 1, X_test_cluster.shape[1]))

        # Model architecture
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    128, return_sequences=True
                ),  # Augmenter le nombre de neurones
                tf.keras.layers.Dropout(0.3),  # Ajuster le dropout
                tf.keras.layers.LSTM(32),  # Ajouter plus de couches
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(
                    32, activation="relu"
                ),  # Couche dense supplémentaire
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Calculate class weights for the specific cluster
        cluster_class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(y_train_cluster),
            y=y_train_cluster.values.ravel(),
        )
        cluster_weights = dict(enumerate(cluster_class_weights))

        # Train model
        model.fit(
            X_train_cluster,
            y_train_cluster,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=cluster_weights,
            verbose=0,
        )

        # Get predictions
        predictions = model.predict(X_test_cluster)
        predictions_binary = (predictions > 0.5).astype(int)

        # Calculate metrics
        balanced_accuracy = balanced_accuracy_score(y_test_cluster, predictions_binary)

        return model, balanced_accuracy

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

                output = pd.concat(
                    [
                        output,
                        pd.DataFrame(
                            {"Classifier": [classifier_name], "Score": [score]}
                        ),
                    ],
                    ignore_index=True,
                )

            except:
                continue

        return output.sort_values(by="Score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    model = Model(
        n_components=0.8,
        cluster_model=ClusterType.kmeans,
        n_cluster=3,
        random_state=42,
        spot_dif=False,
    )