from src.StreamPort.core.Engine import Engine
from src.StreamPort.ml.MachineLearningAnalyses import MachineLearningAnalyses
from src.StreamPort.device.DeviceAnalyses import DeviceAnalyses
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# to run isolation forest and assign classes on its basis
from sklearn.ensemble import IsolationForest as iso

# to split data into training and testing sets
from sklearn.model_selection import train_test_split as splitter
import plotly.graph_objects as go
import plotly.express as px

# from sklearn.metrics import confusion_matrix, classification_report


class MachineLearningEngine(Engine):
    """
    A class for running machine learning that inherits from Engine class.

    Attributes:
        headers (Metadata, optional): The project headers. Instance of Metadata class.
        settings (list, optional): The list of settings. Instance or list of instances of ProcessingMethod class.
        analyses (list, optional): The list of analyses. Instance or list of instances of MachineLearningAnalyses class.
        results (dict, optional): The dictionary of results.
        _anomalies_df (df, optional): DataFrame object containing all detected anomalies for the current object.

    Methods:
        __init__ (self, headers=None, settings=None, analyses=None, results=None): Initializes the Engine instance.
        add_analyses_from_csv (self, path): Reads a CSV file and adds analyses to the engine.
        add_classes_from_csv (self, path): Readas a CSV file and adds classes to the engine.
        get_data (self): Get data array from analyses.
        add_classes (self, classes): Adds classes to each Analyses for classification.
        get_classes (self): Get the added classes.
        make_pca (self): Perform PCA and collects the results.
        plot_pca (self): Plots the PCA results and classes.

    """

    def __init__(
        self,
        headers=None,
        settings=None,
        analyses=None,
        results=None,
        anomalies_df=None,
        curve_data=None,
        features_df=None,
    ):
        """
        Initializes the MachineLearningEngine instance

        Args:
            headers (Metadata, optional): The project headers.
            settings (list, optional): The list of settings.
            analyses (list, optional): The list of analyses.
            results (dict, optional): The dictionary of results.
        """

        super().__init__(headers, settings, analyses, results)
        self._classes = []
        self._anomalies_df = anomalies_df
        self._curve_data = curve_data
        self._features_df = features_df

    def get_analyses(self, analyses=None):
        """
        Identical superclass method is modified here to return only a list,
        and also to return any value with matching subwords rather than the exact key.

        """
        ana_list = self._analyses

        if analyses != None:
            ana_list = []
            if isinstance(analyses, int) and analyses < len(self._analyses):
                ana_list.append(self._analyses[analyses])

            elif isinstance(analyses, str):
                for Analyses in self._analyses:
                    if analyses in Analyses.name:
                        ana_list.append(Analyses)

            elif isinstance(analyses, DeviceAnalyses):
                ana_list.append(analyses)

            elif isinstance(analyses, list):
                analyses_out = []
                for Analyses in analyses:
                    if isinstance(Analyses, int) and Analyses < len(self._analyses):
                        analyses_out.append(self._analyses[Analyses])
                    elif isinstance(Analyses, str):
                        for a in self._analyses:
                            if Analyses in a.name:
                                analyses_out.append(a)
                    elif isinstance(Analyses, DeviceAnalyses):
                        analyses_out.append(Analyses)
                ana_list = analyses_out

            else:
                print("Analyses not found!")

        else:
            print(
                "Provided data is not sufficient or does not exist! Existing analyses will be returned."
            )

        return ana_list

    def add_analyses_from_csv(self, path=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file. (extra details about the csv structure for user)

        Raises:
            ValurError: if the structure of the csv file is not expected
            FileNotFoundError: if the csv file does not exist
        """

        if path is not None:
            if os.path.exists(path):
                df = pd.read_csv(path)
                structure = {
                    "number_of_rows": df.shape[0],
                    "number_of_columns": df.shape[1],
                }

                if (
                    structure["number_of_rows"] == 0
                    or structure["number_of_columns"] == 0
                ):
                    raise ValueError(
                        "The structure of the CSV file is not as expected."
                    )
                else:
                    print(f"Structure of the CSV file: {structure}")
            else:
                raise FileNotFoundError(f"The file {path} does not exist.")
        else:
            return None

        analyses_name = df.iloc[:, 0].tolist()

        if df.duplicated("name", keep="first").any():
            print(
                "Warning: Duplicate Analyses names found in the CSV file. Only the first will be added!"
            )

        column_names = df.columns.tolist()[1:]

        for index, row in df.iterrows():
            row_value = row.tolist()[1:]
            ana = MachineLearningAnalyses(
                name=str(analyses_name[index]),
                data={"x": np.array(column_names), "y": np.array(row_value)},
            )
            if ana.validate():
                self.add_analyses(ana)
            else:
                print(f"Analyses {analyses_name[index]} did not pass validation.")

    def add_classes_from_csv(self, class_path=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file. (extra details about the csv structure for user)

        Raises:
            ValurError: if the structure of the csv file is not expected
            FileNotFoundError: if the csv file does not exist
        """

        if class_path is not None:
            if os.path.exists(class_path):
                df = pd.read_csv(class_path)
                structure = {
                    "number_of_rows": df.shape[0],
                    "number_of_columns": df.shape[1],
                }

                if (
                    structure["number_of_rows"] == 0
                    or structure["number_of_columns"] == 0
                ):
                    raise ValueError(
                        "The structure of the CSV file is not as expected."
                    )
                else:
                    print(f"Structure of the CSV file: {structure}")
            else:
                raise FileNotFoundError(f"The file {class_path} does not exist.")
        else:
            return None

        class_name = df["class"].tolist()
        column_names = df.columns.tolist()[1:]

        for index, row in df.iterrows():
            row_value = row.tolist()[1:]
            ana = MachineLearningAnalyses(
                name=str(class_name[index]),
                data={"x": np.array(column_names), "y": np.array(row_value)},
            )
            if ana.validate():
                self.add_classes(class_name[index])
            else:
                print(f"Analyses {class_name[index]} did not pass validation.")

    def get_data(self):
        """
        Method for collapse all data arrays from analyses into a matrix for statistics

        """

        if not self._analyses:
            print("No analyses found")
            return None

        x_values = self._analyses[0].data["x"]

        matrix = []
        for Analyses in self._analyses:
            y_values = Analyses.data["y"]
            matrix.append(y_values)

        df_matrix = pd.DataFrame(matrix, columns=x_values)

        return df_matrix

    def get_device_data(self, device):
        """
        Retrieves data from linked DeviceEngine object for classification.
        DeviceEngine object returns a list of MLEngine objects with scaled and prepared MLAnalyses objects compatible with MLEngine specifications.

        """
        features_analyses = []
        methods = []
        for method in list(device._method_ids):
            if device.trim_method_name(method) not in methods:
                methods.append(device.trim_method_name(method))
        for resname in methods:
            features_Analyses = device.get_feature_matrix(results=resname)
            features_analyses.append(features_Analyses)
        return (features_analyses, methods)

    def add_classes(self, classes):
        """
        Adds classes (a array of string) to each Analyses and use it for classification of the PCA results

        Args:
            classes(str or list[str]): The classes or list of classes to add.

        Raises:
            TypeError: if the classes parameter is not an instance o a list of instances of string array
            TypeError: if any element in the list of classes is not an instance of string array
        """
        if not self._analyses:
            print("No analyses found")
            return None

        if self._classes is None:
            self._classes = []

        if isinstance(classes, list):
            for class_list in classes:
                if not isinstance(class_list, str):
                    raise TypeError("Each element in the classes list must be a string")
                if class_list not in self._classes:
                    self._classes.append(class_list)
        else:
            if not isinstance(classes, str):
                raise TypeError("The classes must be an instance or a string")
            if classes not in self._classes:
                self._classes.append(classes)

    def get_classes(self):
        """
        Method to get the added classes.
        """

        return self._classes

    def make_model(self):
        # Create a method in the ML engine to perfom PCA and collect the results
        """
        Method to perform plot and collect the results
        """
        if not self._analyses:
            print("No analyses found")
            return None

        # get the settings for PCA from _settings attribute or get_settings from self
        settings = self.get_settings(settings="MakeModel")
        if settings is None:
            print("No setting found")
            return None
        # to find the number of components
        # settings_obj = _settings[which is class MakePCA], return the first
        for settings in self._settings:
            if settings.call == "MakeModel":
                settings_obj = settings
                break
            else:
                settings_obj = None
        # result = settings_obj.run(self)
        if settings_obj:
            result = settings_obj.run(self)
            # self.add_results(result)

            # create a class for model representation
            # add an attribute with model type: PCA, PLS, etc.

            # get data from the model
            # plot data from model
            # predict with model based on new data

            self.add_results({"model": result})
        else:
            print("No settings object found")

    def make_iso_forest(
        self, data=None, curve_data=None, random_state=None, train=None, test=None
    ):
        """
        3-way function that uses ML engines that each run host DeviceEngine methods to classify data from a particular method.
        ML object with an iteration of this function exists for each unique method id in DeviceEngine.

        """
        anomalies = []
        classes = []
        anomalies_df = None

        self._curve_data = (
            curve_data if not isinstance(curve_data, type(None)) else self._curve_data
        )
        self._features_df = (
            data if not isinstance(data, type(None)) else self._features_df
        )

        random_state = random_state
        # handle missing values if any
        data.fillna(0, inplace=True)
        # split data into training and testing sets
        train_data, test_data = splitter(data, test_size=0.5, random_state=random_state)

        print("Diag:", train_data.shape)
        # contamination:
        # Description: This parameter specifies the proportion of outliers in the dataset.
        # Default: 'auto', which estimates the contamination based on the data.

        # bootstrap:
        # Description: If set to True, individual trees are fit on random subsets of the training data sampled with replacement.
        # Default: False.
        # Impact: Bootstrapping can help improve the robustness of the model by introducing more variability in the training data.

        classifier = iso(contamination=0.15, bootstrap=True, random_state=random_state)
        classifier.fit(train_data)

        prediction = classifier.decision_function(test_data)

        # try better threshold than mean. Set as hyperparameter
        # find std for anomaly scores and use as threshold for decision function
        mean_pred = prediction.mean()
        # mean_std = prediction.std()
        mean_std = mean_pred * prediction.std()

        # set outlier detection threshold
        threshold = prediction < mean_std

        # Assign different colors to normal data and anomalies
        colors = np.where(threshold, "red", "black")

        labels = np.where(colors == "red", -1, 0)

        # Assign different sizes to outliers and inliers
        sizes = np.where(threshold, 30, 20)

        test_set = test_data.index
        train_set = train_data.index

        def plot_anomalies(dataset, colors=None):
            test_or_train = 0
            if dataset == "train":
                colors = ["black" for i in range(len(train_set))]
                title = "Training set"
                test_or_train = 1
                dataset = train_set
            else:
                title = "Anomalous curves(Red) - test set"
                dataset = test_set
                colors = colors
            """
            First show prediction w.r.t threshold values
            """
            if test_or_train != 1:
                # Create the scatter plot
                fig = go.Figure()

                for i in range(len(dataset)):
                    fig.add_trace(
                        go.Scatter(
                            x=[dataset[i].split("|")[-1]],
                            y=[prediction[i]],
                            mode="markers",
                            marker=dict(color=colors[i], size=sizes[i]),
                            text=dataset[i],
                            name=dataset[i].split("|")[-1],
                        )
                    )

                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Samples",
                    yaxis_title="Anomaly scores",
                    yaxis=dict(dtick=0.1),  # Set the y-axis resolution to 0.005
                )

                # Show the plot
                fig.show()

            """
            Then show anomalous curves
            """
            # Create the scatter plot
            fig = go.Figure()
            time_axis = curve_data["Time"]

            # loop over all samples in the test set
            for i in range(len(dataset)):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=curve_data[dataset[i]],
                        visible=True,
                        mode="lines",
                        marker=dict(color=colors[i], size=sizes),
                        text=dataset[i],
                        name=dataset[i].split("|")[-1],
                    )
                )
                # get run start date from sample name and use it to find the appropriate Analyses/analyses
                curve_timestamp = dataset[i].split("|")[-1]
                Analyses = self.get_analyses(curve_timestamp)
                # there can only be one Analyses with a unique timestamp
                Analyses = Analyses[0]
                # if current run found to be an anomaly, its respective Analyses object's class is set to indicate it.
                if colors[i] == "red" and Analyses.classes != "Deviant":
                    Analyses.set_class_label("Deviant")
                    anomalies.append(Analyses.name)
                    classes.append(Analyses.classes)
                else:
                    Analyses.set_class_label("Normal")

                self.add_classes(Analyses.classes)

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Pressure",
                # yaxis=dict(
                #    dtick=0.005  # Set the y-axis resolution to 0.05
                # )
            )

            # Show the plot
            fig.show()

        print("Training set")
        plot_anomalies("train")
        print("Test set")
        plot_anomalies("test", colors=colors)

        anomalies_df = pd.DataFrame({"samples": anomalies, "classes": classes})
        self._anomalies_df = anomalies_df

        print("Classes")
        for ana in self._analyses:
            print(f"{ana.name} : {ana.classes}")

        return labels

    def get_anomalies(self):
        return self._anomalies_df

    def plot_data(self):
        """
        Method for general plot of data from the Analyses using Plotly.

        """
        # add argument to optionally choose the x val as rows or cols
        # look at plotly for interactive plotting https://plotly.com/python/

        if not self._analyses:
            print("No analyses found")
            return None

        data = self.get_data()
        if data is None:
            print("No data found")
            return None

        fig = go.Figure()
        for i, Analyses in enumerate(self._analyses):
            fig.add_trace(
                go.Scatter(x=data.columns, y=data.iloc[i], name=Analyses.name)
            )

        fig.update_layout(
            title="General Data", xaxis_title="Feature", yaxis_title="Value"
        )
        fig.write_html("general_data_plot.html")
        # save the control fault

    def plot_pca(self):
        """
        Method to plot the PCA results and classes, including explained variance percentages.
        """
        if not self._analyses:
            print("No analyses found")
            return None

        feature_names = self._analyses[0].data["x"]
        if feature_names is None:
            print("No feature names found")
            return None

        pca_results, pca = self.get_results("pca_model")
        if pca_results is None:
            print("No PCA results found")
            return None

        classes = self.get_classes()
        if classes is None:
            print("No classes found")
            return None

        # PCA Scores DataFrame erstellen
        pca_df = pd.DataFrame(data=pca_results[:2], columns=["PC1", "PC2"])
        if len(classes) != len(pca_df):
            classes = (classes * len(pca_df))[: len(pca_df)]

        pca_df["class"] = classes

        # Ensure 'classes' length matches 'pca_df' rows
        if len(classes) != len(pca_df):
            print(
                "Warning: Number of classes does not match number of data points. Please check class file."
            )
            classes = (classes * (len(pca_df) // len(classes) + 1))[: len(pca_df)]
        pca_df["class"] = classes

        # PCA Scores Plot
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="class",
            title="PCA Scores",
            template="plotly",
        )
        fig.show()

        # PCA Loadings Plot
        loadings = pd.DataFrame(
            pca.components_[:2].T, columns=["PC1", "PC2"], index=feature_names
        )
        fig = px.scatter(
            loadings,
            x="PC1",
            y="PC2",
            text=loadings.index,
            title="PCA Loadings",
            template="plotly",
        )
        fig.update_traces(
            textposition="top center", textfont=dict(size=12), marker=dict(size=10)
        )
        fig.show()

        print("Unique classes in data:", pca_df["class"].unique())
        print("Class distribution:", pca_df["class"].value_counts())

    def plot_dbscan(self):

        if not self._analyses:
            print("No analyses found")
            return None

        dbscan_results = self.get_results("dbscan_model")
        if dbscan_results is None:
            print("No dbscan results found")
            return None

        classes = self.get_classes()
        if classes is None:
            print("No classes found")
            return None

        # Plot the results
        plt.figure(figsize=(10, 8))

        # Unique labels
        fitted_model = dbscan_results[list(dbscan_results.keys())[0]]
        labels = fitted_model.labels_

        # print labes to console
        print("Estimated number of clusters: %d" % len(set(labels)))

        unique_labels = set(labels)
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
        ]

        data = self.get_data()

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = data[class_member_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

            for i in range(len(xy)):
                plt.annotate(classes[class_member_mask][i], (xy[i, 0], xy[i, 1]))

        plt.title("DBSCAN Clustering Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

        # dbscan_comp1 = dbscan_data[:, 0]
        # dbscan_comp2 = dbscan_data[:, 1]

        # unique_labels = np.unique(dbscan_results)
        # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        # for label, color in zip(unique_labels, colors):
        #     if label == -1:

        #         color = "black"

        #     class_member_mask = (dbscan_results == label)
        #     xy = dbscan_data[class_member_mask]
        #     plt.scatter(xy[:, 0], xy[:, 1], color=color, edgecolor='k', label=f'Cluster {label}')

        # plt.scatter(dbscan_comp1, dbscan_comp2, alpha=0.2)
        # plt.xlabel('comp1')
        # plt.ylabel('comp2')
        # plt.title('DBSCAN Clustering')
        # plt.colorbar(label='Cluster Label')
        # plt.show()

    def plot_umap(self):

        if not self._analyses:
            print("No analyses found")
            return None

        umap_results, umap = self.get_results("umap_model")
        if umap_results is None:
            print("No umap results found")
            return None

        classes = self.get_classes()
        if classes is None:
            print("No classes found")
            return None

        # Create a DataFrame for the UMAP results
        umap_df = pd.DataFrame(data=umap_results, columns=["UMAP1", "UMAP2"])
        if len(classes) != len(umap_df):
            classes = (classes * len(umap_df))[: len(umap_df)]

        umap_df["class"] = classes
        fig = px.scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            color="class",
            title="UMAP Projection",
            labels={"UMAP1": "UMAP Component 1", "UMAP2": "UMAP Component 2"},
            template="plotly",
        )
        fig.show()

    def add_month_classes(self, df, month):
        """
        Adds classes for a specific month to the engine.

        Args:
            df: DataFrame containing class data with a 'month' column.
            month: String specifying the month to filter data by.
        """
        if isinstance(month, str):
            month = [month]

        print("plot data for month:", ", ".join(month))

        df_filtered = df[df["month"].isin(month)]

        for index, row in df_filtered.iterrows():
            row_value = row.tolist()[1:]
            class_name = row["monthclass"]
            ana = MachineLearningAnalyses(
                name=str(class_name),
                data={
                    "x": np.array(df_filtered.columns.tolist()[1:]),
                    "y": np.array(row_value),
                },
            )
            if ana.validate():
                self.add_classes(class_name)
            else:
                print(f"Analyses {class_name} did not pass validation.")

    def add_polarity_classes(self, df, polarity):
        """
        Adds classes for a specific month to the engine.

        Args:
            df: DataFrame containing class data with a 'month' column.
            polarity: String specifying the polarity to filter data by ('positive' or 'negative').
        """
        if isinstance(polarity, str):
            polarity = [polarity]

        print(f"plot {polarity} polarity classes")

        df_filtered = df[df["polarity"].isin(polarity)]

        for index, row in df_filtered.iterrows():
            row_value = row.tolist()[1:]
            class_name = row["class"]
            ana = MachineLearningAnalyses(
                name=str(class_name),
                data={
                    "x": np.array(df_filtered.columns.tolist()[1:]),
                    "y": np.array(row_value),
                },
            )
            if ana.validate():
                self.add_classes(class_name)
            else:
                print(f"Analyses {class_name} did not pass validation.")

    # def plot_random_forest(self):

    #     if not self._analyses:
    #         print("No analyses found")
    #         return None

    #     rf_results, rf_model = self.get_results("random_forest_model")
    #     if rf_results is None:
    #         print("No random forest results found")
    #         return None

    #     classes = self.get_classes()
    #     if classes is None:
    #         print("No classes found")
    #         return None

    #     data = self.get_data()
    #     target = self.get_target()

    #     # Vorhersagen treffen
    #     y_pred = rf_model.predict(data)

    #     # Ergebnisse ausgeben
    #     print("Classification Report:\n", classification_report(target, y_pred))
    #     print("Confusion Matrix:\n", confusion_matrix(target, y_pred))

    #     # Plotten der Ergebnisse
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=20)
    #     plt.title('Random Forest Classification Results')
    #     plt.xlabel('Feature 1')
    #     plt.ylabel('Feature 2')
    #     plt.show()
