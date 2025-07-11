import pandas as pd
from src.ingestion.ingestorFactory import IngestorFactory
import streamlit as st


def load_dataset(url: str, file_number=1, save_file=False) -> pd.DataFrame:
    ingestor_factory = IngestorFactory(url, file_number, save_file)
    ingestor = ingestor_factory.create()
    return ingestor.load_data()
