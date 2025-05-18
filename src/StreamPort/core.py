"""
Core module for the StreamPort project.

This module contains the core classes and functionality for managing metadata,
processing methods, workflows, analyses, and the core engine of a project.

Classes:
    Metadata: Represents the metadata of a project, extending the functionality of a dictionary.
    ProcessingMethod: Represents a processing method in a workflow.
    Workflow: Manages an ordered list of ProcessingMethod objects.
    Analyses: Represents the analyses performed in a project.
    Engine: Manages the core components of a project.

Dependencies:
    - datetime: For handling date and time information.
    - pandas: For creating and managing data frames.
"""

import datetime
import pandas as pd


class Metadata(dict):
    """
    Represents the metadata of a project, extending the functionality of a dictionary.

    Methods:
      validate(): Validates the project metadata.
      __str__(): Returns a string representation of the project metadata.
      print(): Prints the project metadata.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Metadata class.

        Args:
            **kwargs: Keyword arguments representing the project metadata.
                The default keys are 'name', 'author', 'path', and 'date'.
        """
        defaults = {
            "name": None,
            "author": None,
            "path": None,
            "date": datetime.datetime.now(),
        }
        super().__init__(defaults)  # Initialize the dict with default values
        self.update(kwargs)  # Update with any provided keyword arguments

    def validate(self):
        """
        Validates the project metadata.

        Raises:
            ValueError: If any of the required fields are missing or invalid.
        """
        required_fields = ["name", "author", "path"]
        for field in required_fields:
            if field not in self or not self[field]:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(self["name"], str):
            raise ValueError("Project name must be a string")
        if not isinstance(self["author"], str):
            raise ValueError("Author name must be a string")
        if not isinstance(self["path"], str):
            raise ValueError("Project path must be a string")
        if not isinstance(self["date"], (str, datetime.datetime)):
            raise ValueError("Date must be a string or datetime object")

    def __str__(self):
        """
        Returns a string representation of the project metadata.

        Returns:
            str: A string representation of the project metadata.
        """
        return "\nMetadata\n" + "\n".join(
            [f"  {key}: {value}" for key, value in self.items()]
        )


class ProcessingMethod:
    """
    ProcessingMethod class to represent a processing method in a workflow.
    """

    def __init__(
        self,
        data_type="",
        method="",
        required="",
        number_permitted=1,
        algorithm="",
        input_instance="",
        output_instance="",
        parameters=None,
        version="",
        software="",
        developer="",
        contact="",
        link="",
        doi="",
    ):
        self.data_type = str(data_type)
        self.method = str(method)
        self.required = str(required)
        self.number_permitted = int(number_permitted)
        self.algorithm = str(algorithm)
        self.input_instance = str(input_instance)
        self.output_instance = str(output_instance)
        self.parameters = dict(parameters) if parameters else {}
        self.version = str(version)
        self.software = str(software)
        self.developer = str(developer)
        self.contact = str(contact)
        self.link = str(link)
        self.doi = str(doi)

    def validate(self):
        """
        Validates the ProcessingMethod object.
        Raises:
            ValueError: If any validation checks fail.
        """
        if not isinstance(self.data_type, str):
            raise ValueError("data_type must be a string")
        if not isinstance(self.method, str):
            raise ValueError("method must be a string")
        if not isinstance(self.required, str):
            raise ValueError("required must be a string")
        if not isinstance(self.number_permitted, int):
            raise ValueError("number_permitted must be an integer")
        if not isinstance(self.algorithm, str):
            raise ValueError("algorithm must be a string")
        if not isinstance(self.input_instance, str):
            raise ValueError("input must be a string")
        if not isinstance(self.output_instance, str):
            raise ValueError("output must be a string")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dictionary")
        if not isinstance(self.version, str):
            raise ValueError("version must be a string")
        if not isinstance(self.software, str):
            raise ValueError("software must be a string")
        if not isinstance(self.developer, str):
            raise ValueError("developer must be a string")
        if not isinstance(self.contact, str):
            raise ValueError("contact must be a string")
        if not isinstance(self.link, str):
            raise ValueError("link must be a string")
        if not isinstance(self.doi, str):
            raise ValueError("doi must be a string")

    def __str__(self):
        result = "\n"
        result += "ProcessingMethod\n"
        result += "  data_type    " + str(self.data_type) + "\n"
        result += "  method       " + str(self.method) + "\n"
        result += "  required     " + str(self.required) + "\n"
        result += "  algorithm    " + str(self.algorithm) + "\n"
        result += "  input        " + str(self.input_instance) + "\n"
        result += "  output       " + str(self.output_instance) + "\n"
        result += "  version      " + str(self.version) + "\n"
        result += "  software     " + str(self.software) + "\n"
        result += "  developer    " + str(self.developer) + "\n"
        result += "  contact      " + str(self.contact) + "\n"
        result += "  link         " + str(self.link) + "\n"
        result += "  doi          " + str(self.doi) + "\n"
        result += "\n"
        result += "  parameters: "
        if len(self.parameters) == 0:
            result += "empty " + "\n"
        else:
            result += "\n"
            for i, param in enumerate(self.parameters):
                if isinstance(param, pd.DataFrame):
                    result += (
                        "  - "
                        + str(list(self.parameters.keys())[i])
                        + " (only head rows)"
                        + "\n"
                    )
                    result += "\n"
                    result += str(param.head()) + "\n"
                    result += "\n"
                elif isinstance(param, dict):
                    result += (
                        "  - " + str(list(self.parameters.keys())[i]) + ": " + "\n"
                    )
                    for key, value in param.items():
                        result += f"      - {key}{value}\n"
                elif callable(param):
                    result += "  - " + str(list(self.parameters.keys())[i]) + ":\n"
                    result += str(param) + "\n"
                else:
                    result += (
                        "  - "
                        + str(list(self.parameters.keys())[i])
                        + str(param)
                        + "\n"
                    )
        return result

    def run(self, analyses):
        """
        Runs the processing method on the provided analyses.

        Args:
            analyses (Analyses): The Analyses object to process.

        Returns:
            Analyses: The updated Analyses object.
        """
        # Placeholder for actual processing logic
        print(f"Running {self.method} with {self.algorithm}...")
        # Here you would implement the actual processing logic
        return analyses


class Workflow(list):
    """
    Workflow class to manage an ordered list of ProcessingMethod objects.
    """

    def __init__(self, processing_methods=None):
        """
        Initializes a Workflow object.

        Args:
            processing_methods (list): A list of ProcessingMethod objects.
        """
        if processing_methods is None:
            processing_methods = []
        elif not isinstance(processing_methods, list):
            raise ValueError("processing_methods must be a list!")
        elif not all(isinstance(pm, ProcessingMethod) for pm in processing_methods):
            raise ValueError(
                "All items in processing_methods must be instances of ProcessingMethod!"
            )

        super().__init__(processing_methods)

    @property
    def methods(self):
        """
        Returns a list of method identifiers for each ProcessingMethod in the workflow.

        Returns:
            list: A list of method identifiers.
        """
        return [f"{pm.data_type}Method_{pm.method}_{pm.algorithm}" for pm in self]

    @property
    def overview(self):
        """
        Returns a DataFrame with an overview of the processing methods in the workflow.

        Returns:
            pd.DataFrame: A DataFrame containing details of the processing methods.
        """
        if len(self) == 0:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "index": range(1, len(self) + 1),
                "method": [pm.method for pm in self],
                "algorithm": [pm.algorithm for pm in self],
                "number_permitted": [pm.number_permitted for pm in self],
                "version": [pm.version for pm in self],
                "software": [pm.software for pm in self],
                "developer": [pm.developer for pm in self],
                "contact": [pm.contact for pm in self],
                "link": [pm.link for pm in self],
                "doi": [pm.doi for pm in self],
            }
        )

    @property
    def data_type(self):
        """
        Returns the data type of the ProcessingMethod objects in the workflow.

        Returns:
            str: The data type of the ProcessingMethod objects.
        """
        if len(self) == 0:
            return None
        return self[0].data_type

    def validate(self):
        """
        Validates the Workflow object.

        Raises:
            ValueError: If any validation checks fail.
        """
        if len(self) == 0:
            return

        # Ensure all items are ProcessingMethod objects
        if not all(isinstance(pm, ProcessingMethod) for pm in self):
            raise ValueError(
                "All items in the workflow must be instances of ProcessingMethod!"
            )

        # Ensure all ProcessingMethod objects have the same data_type
        data_types = {pm.data_type for pm in self}
        if len(data_types) > 1:
            raise ValueError(
                "All ProcessingMethod objects must have the same data_type!"
            )

        # Ensure methods with number_permitted == 1 are unique
        unique_methods = [pm.method for pm in self if pm.number_permitted == 1]
        if len(unique_methods) != len(set(unique_methods)):
            raise ValueError(
                "All ProcessingMethod objects with number_permitted == 1 must have unique methods!"
            )

    def __str__(self):
        """
        Returns a string representation of the Workflow.

        Returns:
            str: A string representation of the Workflow.
        """
        if len(self) == 0:
            return "\nWorkflow\n  No processing methods in the workflow."

        return "\nWorkflow\n" + "\n".join(
            [f"  {i + 1}. {pm.method} ({pm.algorithm})" for i, pm in enumerate(self)]
        )


class Analyses:
    """
    Analyses class to represent data (including results) in a project.
    """

    def __init__(self, data_type: str = "", formats: list = None, data: list = None):
        self.data_type = str(data_type) if data_type else ""
        self.formats = formats if formats is not None else []
        self.data = data if data is not None else []
        self.validate()

    def validate(self):
        """
        Validates the Analyses object.
        Raises:
            ValueError: If any validation checks fail.
        """

        if not isinstance(self.data_type, str):
            raise ValueError("data_type must be a string")
        if not isinstance(self.formats, list):
            raise ValueError("formats must be a list")
        if not all(isinstance(item, str) for item in self.formats):
            raise ValueError("formats must be a list of strings")
        if not isinstance(self.data, list):
            raise ValueError("data must be a list")

    def __str__(self):
        """
        Returns a string representation of the Analyses instance.
        """
        str_data = ""
        if len(self.data) > 0:
            for i, item in enumerate(self.data):
                if isinstance(item, pd.DataFrame):
                    str_data += (
                        f"    {i + 1}. {item.shape[0]} rows x {item.shape[1]} columns\n"
                    )
                elif isinstance(item, dict):
                    str_data += f"    {i + 1}. {len(item)} items\n"
                else:
                    str_data += f"    {i + 1}. {str(item)}\n"
        else:
            str_data += "  No data available."

        return (
            f"\n{type(self).__name__} \n"
            f"  data_type: {self.data_type} \n"
            f"  formats: {self.formats} \n"
            f"  data: {len(self.data)} \n"
            f"{str_data} \n"
        )


class Engine:
    """
    Engine class to manage the a data processing project for a defined type analyses data.
    """

    _metadata = None
    _analyses = None
    _workflow = None

    def __init__(self, metadata=None, analyses=None, workflow=None):
        self._metadata = Metadata()
        self._analyses = Analyses()
        self._workflow = Workflow()

        if metadata is not None:
            if isinstance(metadata, Metadata):
                metadata.validate()
                self.metadata = metadata

        if analyses is not None:
            if isinstance(analyses, Analyses):
                analyses.validate()
                self.analyses = analyses

        if workflow is not None:
            if isinstance(workflow, Workflow):
                workflow.validate()
                self.workflow = workflow

    @property
    def metadata(self):
        """
        Returns the metadata of the Engine instance.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        if isinstance(value, Metadata):
            value.validate()
            self._metadata = value
        else:
            raise TypeError("metadata must be an instance of Metadata")

    @property
    def analyses(self):
        """
        Returns the analyses of the Engine instance.
        """
        return self._analyses

    @analyses.setter
    def analyses(self, value):
        if isinstance(value, Analyses):
            value.validate()
            self._analyses = value
        else:
            raise TypeError("analyses must be an instance of Analyses")

    @property
    def workflow(self):
        """
        Returns the workflow of the Engine instance.
        """
        return self._workflow

    @workflow.setter
    def workflow(self, value):
        if isinstance(value, Workflow):
            value.validate()
            self._workflow = value
        else:
            raise TypeError("workflow must be an instance of Workflow")

    def __str__(self):
        """
        Returns a string representation of the Engine instance.
        """
        return (
            f"\n{type(self).__name__} \n"
            f"  name: {self._metadata['name']} \n"
            f"  author: {self._metadata['author']} \n"
            f"  path: {self._metadata['path']} \n"
            f"  date: {self._metadata['date']} \n"
        )

    def print(self):
        """
        Prints the Engine instance.
        """
        print(self)
        print(self.workflow)
        print(self.analyses)

    def run(self):
        """
        Runs the workflow in the engine adding results to the analyses.
        """
        print("Running the workflow")

        number_of_methods = len(self.workflow)
        if number_of_methods == 0:
            print("No methods in the workflow.")
            return

        processed_methods = 1

        for method in self.workflow:
            print(
                f"Processing method {type(method).__name__} ({processed_methods} / {number_of_methods})"
            )
            processed_methods += 1
            self.analyses = method.run(self.analyses)
