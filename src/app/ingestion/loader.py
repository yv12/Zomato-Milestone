from datasets import load_dataset
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str = "ManikaSaini/zomato-restaurant-recommendation"):
        self.dataset_name = dataset_name

    def fetch_as_dataframe(self) -> pd.DataFrame:
        print(f"Fetching dataset '{self.dataset_name}' from Hugging Face...")
        dataset = load_dataset(self.dataset_name)
        
        # Zomato dataset typically has everything in the 'train' split
        df = dataset['train'].to_pandas()
        print(f"Successfully loaded {len(df)} raw rows.")
        return df
