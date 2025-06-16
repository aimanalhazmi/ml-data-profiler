import pandas as pd
import numpy as np
from datasets import load_dataset #Todo integrate into requirements.txt V 3.6.0
from huggingface_hub import DatasetInfo, HfApi
from kaggle.api.kaggle_api_extended import KaggleApi #ToDo integrate into requirements kaggle-1.7.4.5
import os

"""

compatible with:
- Tabular data
- HuggingFace:   - csv, json, parquet           IN repositories with multiple files takes first file with fitting format
- Kaggle: - csv, json, parquet  NOT SQLlite

"""

class BaseIngestor:
    def __init__(self, link: str, file_index: int):
        self.link = link
        self.file_index = file_index

    def load_data(self):
        raise NotImplementedError

class IngestorFactory:
    def __init__(self, link: str, file_number: int = 0):
        self.link = link

        self.file_index = file_number - 1

    def create(self) -> BaseIngestor:
        """
        Create and return the appropriate data ingestor instance based on the given dataset link.

        Returns:
            BaseIngestor: An instance of a subclass of BaseIngestor suitable for the detected platform.

        Raises:
            ValueError: If the link does not belong to a supported platform (Hugging Face, Kaggle, OpenML).
        """

        if "huggingface.co" in self.link:
            return HuggingFaceIngestor(self.link, self.file_index)
        elif "kaggle.com" in self.link:
            return KaggleIngestor(self.link, self.file_index)
        elif "openml.org" in self.link:
            return OpenMLIngestor(self.link, self.file_index)
        else:
            raise ValueError("Unknown platform.")

# load data from hugging face
class HuggingFaceIngestor(BaseIngestor):

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet"
    }

    def __init__(self, link, file_index):
        """
        Initialize the ingestor with a Hugging Face dataset link.

        Args:
            link (str): URL to the Hugging Face dataset page.
            file_index (int): URL to the Hugging Face dataset page.
        """
        super().__init__(link, file_index)

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
        for sibling in repo_info.siblings:
            filename = sibling.rfilename
            for format in self.SUPPORTED_FORMATS:
                if filename.endswith(format):
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
    
    def load_data(self) -> None:
        """
        Load the dataset from Hugging Face, convert it to CSV format, and save it locally.
        The data is saved in the '../../data/' directory using the original file name.
        """
        repo_id = self.get_repo_id()
        repo_info = self.get_repo_info(repo_id)
        file_link, file_format = self.get_file_url(repo_id,repo_info)


        dataset = load_dataset(file_format, data_files=file_link)
        dataset = dataset["train"]

        dataset_name = self.get_name(self.link)      

        save_path = f"../../data/{dataset_name}.csv"
        dataset.to_csv(save_path)

# load data from Kaggle
class KaggleIngestor(BaseIngestor):
    def __init__(self, link, file_index):
        """
        Initialize the Kaggle ingestor with a dataset link and file index.

        Args:
            link (str): URL to the Kaggle dataset.
            file_index (int): Index of the file to be loaded (0-based).
        """
        super().__init__(link, file_index)

    def is_kaggle_configured(self): # ToDO in ReadMe integrieren
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

        if self.file_index > len(file_list):
            raise IndexError(
                 f"file_index {self.file_index} is out of range. Dataset '{dataset_id}' contains only {len(file_list)} files."
            )
    
    def get_dataset_id(self, link):
        """
        Extract the dataset ID from a Kaggle dataset URL.

        Args:
            link (str): The full Kaggle dataset URL.

        Returns:
            str: Dataset ID in the format "owner/dataset".
        """
        relevant_part = link.split("/datasets/")[1].split("/")
        return f"{relevant_part[0]}/{relevant_part[1]}"
    
    def download_kaggle_dataset(self, dataset_id: str, target_dir: str) -> None:
        """
        Download a specific file from a Kaggle dataset to a target directory.

        Args:
            dataset_id (str): The Kaggle dataset ID.
            target_dir (str): Local directory to store the downloaded file.

        Raises:
            RuntimeError: If the Kaggle API key is not found or configured.
        """
        if not self.is_kaggle_configured():
            raise RuntimeError(
                "Kaggle API key not found. Please save `kaggle.json` in ~/.kaggle."
            )
        api = KaggleApi()
        api.authenticate()

        file_name = self.get_name(dataset_id, self.file_index)
        api.dataset_download_file(dataset_id, file_name=file_name, path=target_dir)

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
        return(api.dataset_list_files(dataset_id).files[file_index].name)
        
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
        path = path + dataset_name

        ext = os.path.splitext(path)[1].lower()

        output_path = output_path + dataset_name.replace(ext, "") + ".csv"

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == "parquet":
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
                
    def load_data(self):
        """
        Orchestrate the full data ingestion process:
        - Parse dataset ID
        - Validate file index
        - Download selected file
        - Convert to CSV
        - Remove raw file
        """
        path = "../../data/kaggle_temp/" 
        output_path = "../../data/" 
        dataset_id = self.get_dataset_id(self.link)
        self.download_kaggle_dataset(dataset_id, path)
        dataset_name = self.get_name(dataset_id, self.file_index)
        self.transform_raw_data(path, output_path, dataset_name)
        self.delete_raw_data(path, dataset_name)

# load data from OpenML
class OpenMLIngestor(BaseIngestor):
    def __init__(self, link, file_index):
        super().__init__(link, file_index)
        

    def load_data(self):
        pass
        