import pandas as pd
from pandas.errors import ParserError
import numpy as np
from datasets import load_dataset 
from huggingface_hub import DatasetInfo, HfApi
from kaggle.api.kaggle_api_extended import KaggleApi 
import os
import openml
import chardet
import shutil
import csv

"""

compatible with:
- Tabular data
- HuggingFace:   - csv, json, parquet           IN repositories with multiple files takes first file with fitting format
- Kaggle: - csv, json, parquet  NOT SQLlite

"""

class BaseIngestor:
    def __init__(self, link: str, file_index: int, save_file: bool = False):
        """
        Base class for dataset ingestion.

        Args:
            link (str): URL or identifier for the dataset.
            file_index (int): Zero-based index of which file to load (if multiple exist).
            save_file (bool): Whether to keep the downloaded file locally.
        """
        self.link = link
        self.file_index = file_index
        self.save_file = save_file

    def load_data(self):
        raise NotImplementedError

class IngestorFactory:
    def __init__(self, link: str, file_number: int = 0, save_file: bool = False):
        """
        Factory to choose the correct ingestor based on URL.

        Args:
            link (str): Dataset URL.
            file_number (int): 1-based file index, converted to zero-based internally.
            save_file (bool): Whether to keep the downloaded file on disk.
        """
        self.link = link
        self.file_index = file_number - 1
        self.save_file = save_file

    def create(self) -> BaseIngestor:
        """
        Create and return the appropriate data ingestor instance based on the given dataset link.

        Returns:
            BaseIngestor: An instance of a subclass of BaseIngestor suitable for the detected platform.

        Raises:
            ValueError: If the link does not belong to a supported platform (Hugging Face, Kaggle, OpenML).
        """

        if "huggingface.co" in self.link:
            return HuggingFaceIngestor(self.link, self.file_index, self.save_file)
        elif "kaggle.com" in self.link:
            return KaggleIngestor(self.link, self.file_index, self.save_file)
        elif "openml.org" in self.link:
            return OpenMLIngestor(self.link, self.file_index, self.save_file)
        else:
            raise ValueError("Unknown platform.")


class HuggingFaceIngestor(BaseIngestor):

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet"
    }

    def __init__(self, link, file_index, save_file):
        """
        Initialize the ingestor with a Hugging Face dataset link.

        Args:
            link (str): URL to the Hugging Face dataset page.
            file_index (int): Not used for Hugging Face; present for API consistency.
            save_file (bool): Whether to retain downloaded files locally.
        """
        super().__init__(link, file_index, save_file)

    def get_repo_id(self) -> str:
        """
        Extract the Hugging Face dataset repository ID from the URL.

        Returns:
            str: The repository ID in the format 'owner/dataset_name'.
        """
        path = self.link.split("/datasets/")[-1]
        repo_id = path.split("/")[0] + "/" + path.split("/")[1]
        return repo_id
    
    def get_repo_info(self, repo_id: str) -> DatasetInfo:
        """
        Fetch metadata for the given dataset repository.

        Args:
            repo_id (str): The Hugging Face repository ID.

        Returns:
            DatasetInfo: Metadata including file list, license, etc.
        """

        api = HfApi()
        info = api.dataset_info(repo_id)
        return(info)
    
    def get_file_url(self, repo_id: str, repo_info) -> tuple[str, str]:
        """
        Find a supported file in the repository and return its download URL and format.

        Args:
            repo_id (str): The repository ID.
            repo_info (DatasetInfo): Repository metadata.

        Returns:
            tuple[str, str]: Tuple of (file_url, file_format), where file_format is 'csv', 'json', or 'parquet'.
        """
        # Iterate over the files listed in the dataset metadata
        for sibling in repo_info.siblings:
            filename = sibling.rfilename
            # Check if the file has one of the supported extensions
            for format in self.SUPPORTED_FORMATS:
                if filename.endswith(format):
                    # Construct the direct download URL
                    file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
                    return file_url, self.SUPPORTED_FORMATS[format]
        raise FileNotFoundError("Found no fitting file")
    
    def get_name(self, file_link: str) -> str:
        """
        Extract the file name from the download URL.

        Args:
            file_link (str): Full URL to the file.

        Returns:
            str: File name (including extension).
        """
        file_name = file_link.split("/")[-1]
        return file_name
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from Hugging Face, convert it to CSV format, and optionally save it locally.
        The data is saved in the '../../data/' directory using the original file name.

        Returns:
            pd.DataFrame: The full dataset loaded into a DataFrame.
        """
        # Step 1: determine the Hugging Face repo identifier
        repo_id = self.get_repo_id()
        # Step 2: retrieve metadata (including file list)
        repo_info = self.get_repo_info(repo_id)
        # Step 3: locate a supported file and its format
        file_link, file_format = self.get_file_url(repo_id,repo_info)

        # Step 4: use the datasets library to load the specified file
        dataset = load_dataset(file_format, data_files=file_link)
        # Assume a 'train' split by default
        dataset = dataset["train"]
        # Step 5: derive a local name and write to CSV
        dataset_name = self.get_name(self.link)      
        save_path = f"../../data/{dataset_name}.csv"
        dataset.to_csv(save_path)

        # Read back into pandas for uniform downstream handling
        df = pd.read_csv(save_path)
        # Step 6: remove the temporary file if not requested to keep it
        if not self.save_file:
            os.remove(save_path)
        return df

# load data from Kaggle
class KaggleIngestor(BaseIngestor):
    def __init__(self, link, file_index, save_file):
        """
        Initialize the Kaggle ingestor with a dataset link and file index.

        Args:
            link (str): URL to the Kaggle dataset.
            file_index (int): Index of the file to be loaded (0-based).
            save_file (bool): Whether to keep downloaded file.
        """
        super().__init__(link, file_index, save_file)

    def is_kaggle_configured(self): 
        """
        Check if the Kaggle API is properly configured.

        Returns:
            bool: True if the API key exists in ~/.kaggle/kaggle.json, False otherwise.
        """
        return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    
    def index_range_check(self, dataset_id: str) -> None:
        """
        Ensure that the provided file index is within the available file range.

        Args:
            dataset_id (str): The Kaggle dataset identifier (e.g., "owner/dataset").

        Raises:
            IndexError: If the file index is out of bounds.
        """
        api = KaggleApi()
        api.authenticate()
        file_list = api.dataset_list_files(dataset_id).files

        # Check that the file_index is within [0, len(file_list)-1]
        if self.file_index > len(file_list):
            raise IndexError(
                 f"file_index {self.file_index} is out of range. Dataset '{dataset_id}' contains only {len(file_list)} files."
            )
    
    def get_dataset_id(self, link:str) -> str:
        """
        Extract the dataset ID from a Kaggle dataset URL.

        Args:
            link (str): The full Kaggle dataset URL.

        Returns:
            str: Dataset ID in the format "owner/dataset".
        """
        relevant_part = link.split("/datasets/")[1].split("/")
        return f"{relevant_part[0]}/{relevant_part[1]}"
    
    def download_kaggle_dataset(self, dataset_id: str, path: str) -> None:
        """
        Download a specific file from a Kaggle dataset to a target directory.

        Args:
            dataset_id (str): The Kaggle dataset ID.
            path (str): Local directory to store the downloaded file.

        Raises:
            RuntimeError: If the Kaggle API key is not found or configured.
        """
        # Ensure Kaggle credentials are present
        if not self.is_kaggle_configured():
            raise RuntimeError(
                "Kaggle API key not found. Please save `kaggle.json` in ~/.kaggle."
            )
        # Authenticate with the Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Determine the filename based on the dataset ID and desired index
        file_name = self.get_name(dataset_id, self.file_index)

        # Download and unzip all files in the dataset to the target path
        api.dataset_download_files(dataset_id, path=path, unzip=True)

        # Search through the unzipped directory structure for the target file
        src_path = None
        for root, dirs, files in os.walk(path):
            if file_name in files:
                src_path = os.path.join(root, file_name)
                break

        # If the requested file wasn’t found, raise an error
        if src_path is None:
            raise FileNotFoundError(f"Could not find {file_name} after unzip in {path}")
        
        # Move the file to the top level of `path` if it’s nested in subfolders
        dst_path = os.path.join(path, file_name)
        if src_path != dst_path:
            shutil.move(src_path, dst_path)

        # Remove all other files and directories, leaving only the desired file
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if entry == file_name:
                continue
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)

    def get_name(self, dataset_id: str, file_index: int) -> str:
        """
        Retrieve the filename of a dataset file by index.

        Args:
            dataset_id (str): The Kaggle dataset ID.
            file_index (int): Index of the file in the dataset.

        Returns:
            str: The name of the selected file.
        """
        api = KaggleApi()
        api.authenticate()
        raw_name = api.dataset_list_files(dataset_id).files[file_index].name
        return os.path.basename(raw_name)
        
    def transform_raw_data(self, path: str, output_path: str, dataset_name: str) -> None:
        """
        Load the raw dataset file, convert it if necessary, and save it as a CSV.

        Args:
            path (str): Path to the input file.
            output_path (str): Directory to store the processed CSV.
            dataset_name (str): Original name of the dataset file.

        Raises:
            ValueError: If the file format is unsupported.
        """
        # Construct full input and output file paths
        path = path + dataset_name
        ext = os.path.splitext(path)[1].lower()
        output_path = output_path + dataset_name.replace(ext, "") + ".csv"

        # Handle delimited text formats, attempting to detect encoding and delimiter
        if ext == ".csv" or ext == ".tsv":
            with open(path, "rb") as f:
                sample_bytes = f.read(10_000)
            detected = chardet.detect(sample_bytes)["encoding"] or "utf-8"
            sample_text = sample_bytes.decode(detected, errors="replace")
            try:
                # Use Sniffer to guess the delimiter
                dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
                sep = dialect.delimiter
            except csv.Error:
                # Fallback heuristic: choose semicolon if more frequent than comma
                sep = ";" if sample_text.count(";") > sample_text.count(",") else ","
            df = pd.read_csv(path,
                            sep=sep,
                            encoding=detected,
                            engine="python",          
                            encoding_errors="replace")
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".xlsx":
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Not supported format: {ext}")

        df.to_csv(output_path, index=False)

    def delete_raw_data(self, path: str, dataset_name: str) -> None:
        """
        Delete the original raw dataset file after processing.

        Args:
            path (str): Directory containing the file.
            dataset_name (str): Name of the file to delete.
        """
        path = path + dataset_name

        if os.path.exists(path):
            os.remove(path)
            print(f"{path} was deleted.")
        else:
            print(f"{path} doesn't exist.")
                
    def load_data(self) -> pd.DataFrame:
        """
        Orchestrate the full data ingestion process:
        - Parse dataset ID
        - Validate file index
        - Download selected file
        - Convert to CSV
        - Remove raw file
        """
        # Define temporary and output directories
        path = "../../data/kaggle_temp/" 
        output_path = "../../data/" 

        # Step 1: determine the Kaggle dataset identifier
        dataset_id = self.get_dataset_id(self.link)

        # Step 2: download the specified file into the temp directory
        self.download_kaggle_dataset(dataset_id, path)

        # Step 3: derive the raw filename and convert it to CSV
        dataset_name = self.get_name(dataset_id, self.file_index)
        self.transform_raw_data(path, output_path, dataset_name)

        # Step 4: remove the raw file after conversion
        self.delete_raw_data(path, dataset_name)
        # Construct the path to the final CSV
        final_csv = output_path + dataset_name.replace(os.path.splitext(dataset_name)[1], "") + ".csv"
        # Load the CSV into pandas
        df = pd.read_csv(final_csv)
        if not self.save_file:
            os.remove(final_csv)
        return df

# load data from OpenML
class OpenMLIngestor(BaseIngestor):
    def __init__(self, link, file_index, save_file):
        """
        Initialize the OpenML ingestor with a dataset link.

        Args:
            link (str): URL pointing to an OpenML dataset (must include 'id=...').
            file_index (int): Not used for OpenML, included for API consistency.
            save_file (bool): Whether to keep saved CSV locally.
        """
        super().__init__(link, file_index, save_file)
        
    def get_dataset_id(self, link: str) -> str:
        """
        Extract the dataset ID from the OpenML URL.

        Args:
            link (str): The OpenML dataset URL.

        Returns:
            str: Dataset ID as a string.

        Raises:
            ValueError: If the dataset ID is missing in the link.
        """
        link_as_list = link.split("www.openml.org/")[1].split("&")
        dataset_id = ""
        contains_id = False
        for element in link_as_list:
            if "id=" in element:
                contains_id = True
                dataset_id = element.split("=")[1]
        if not contains_id:
            raise ValueError(
                "The link is not valid, it's missing the id of the dataset."
            )
        return dataset_id

    def add_target_column(self, df, y, dataset):
        """
        Append the target column to the DataFrame using the correct name.

        Args:
            df (pd.DataFrame): The feature data.
            y (pd.Series): Target values.
            dataset (openml.datasets.OpenMLDataset): The OpenML dataset object.

        Returns:
            pd.DataFrame: DataFrame with the target column added.
        """
        target_name = dataset.default_target_attribute
        df[target_name] = y
        return df

    def load_data(self):
        """
        Load a dataset from OpenML, attach the target column if applicable,
        and save it as a CSV file in the local data directory.
        """
        # Step 1: parse the numeric OpenML dataset ID from URL
        dataset_id = self.get_dataset_id(self.link)

        # Step 2: fetch the dataset object and its raw data
        dataset = openml.datasets.get_dataset(dataset_id)
        dataset_name = dataset.name
        df, y, *_ = dataset.get_data()

        # Step 3: if a target vector was returned, merge it into the DataFrame
        if y is not None:
            df = self.add_target_column(df, y, dataset)

        # Sanitize dataset name for filesystem use
        dataset_name = dataset.name.replace(" ", "_")
        save_path = f"../../data/{dataset_name}.csv"

        # Step 4: write out to CSV and reload for consistency
        df.to_csv(save_path, index=False)
        df = pd.read_csv(save_path)

        # Step 5: remove the saved CSV if not requested to persist it
        if not self.save_file:
            os.remove(save_path)
        return df
