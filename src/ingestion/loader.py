import pandas as pd
from src.ingestion.ingestorFactory import IngestorFactory


def load_dataset(url: str, save_file=False) -> pd.DataFrame:
    ingestor_factory = IngestorFactory(url, 1, save_file)
    ingestor = ingestor_factory.create()
    return ingestor.load_data()
