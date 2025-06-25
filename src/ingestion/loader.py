import pandas as pd
from src.ingestion.ingestorFactory import IngestorFactory


def load_dataset(url:str) -> pd.DataFrame:
    ingestor_factory = IngestorFactory(url, 1)
    ingestor = ingestor_factory.create()
    return ingestor.load_data()