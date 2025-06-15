import pandas as pd
import numpy as np
from datasets import load_dataset #Todo integrate into requirements.txt V 3.6.0
from huggingface_hub import DatasetInfo, HfApi

"""
# should cleaning of the data happen here????

compatible with:
- Tabular data
- HuggingFace:   - csv, json, parquet

"""

class BaseIngestor:
    def __init__(self, link: str):
        self.link = link

    def load_data(self):
        raise NotImplementedError

class IngestorFactory:
    def __init__(self, link: str):
        self.link = link

    def create(self) -> BaseIngestor:
        """
        Create and return the appropriate data ingestor instance based on the given dataset link.

        Returns:
            BaseIngestor: An instance of a subclass of BaseIngestor suitable for the detected platform.

        Raises:
            ValueError: If the link does not belong to a supported platform (Hugging Face, Kaggle, OpenML).
        """

        if "huggingface.co" in self.link:
            return HuggingFaceIngestor(self.link)
        elif "kaggle.com" in self.link:
            return KaggleIngestor(self.link)
        elif "openml.org" in self.link:
            return OpenMLIngestor(self.link)
        else:
            raise ValueError("Unknown platform.")

# load data from hugging face
class HuggingFaceIngestor(BaseIngestor):

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet"
    }

    def __init__(self, link):
        """
        Initialize the ingestor with a Hugging Face dataset link.

        Args:
            link (str): URL to the Hugging Face dataset page.
        """
        super().__init__(link)

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

        Raises:
            FileNotFoundError: If no supported file format is found in the repository.
        """
        for sibling in repo_info.siblings:
            filename = sibling.rfilename
            for format in self.SUPPORTED_FORMATS:
                if filename.endswith(format):
                    file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
                    return file_url, self.SUPPORTED_FORMATS[format]
        raise FileNotFoundError("Found no fitting csv file")
    
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
    def __init__(self, link):
        super().__init__(link)

    def load_data(self):
        pass

# load data from OpenML
class OpenMLIngestor(BaseIngestor):
    def __init__(self, link):
        super().__init__(link)

    def load_data(self):
        pass
        