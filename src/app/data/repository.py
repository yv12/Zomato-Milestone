import os
import sqlite3
import numpy as np
import pandas as pd
from typing import List
from app.config import settings
from ..models.domain import Restaurant, FilterCriteria

class RestaurantRepository:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or settings.absolute_data_path
        self._df = None
        self._load_data()

    def _load_data(self):
        """Loads and parses data from SQLite into an in-memory Pandas DataFrame."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data store not found at {self.data_path}. Please run the data ingestion pipeline first."
            )
            
        print(f"Connecting to SQLite data store at {self.data_path}...")
        conn = sqlite3.connect(self.data_path)
        
        # Load all rows into a dataframe
        df = pd.read_sql_query("SELECT * FROM restaurants", conn)
        conn.close()
        
        # De-serialize comma-separated strings back into lists for cuisines and dish_liked
        df['cuisines'] = df['cuisines'].fillna('').astype(str).apply(
            lambda x: [c.strip() for c in x.split(',') if c.strip()]
        )
        df['dish_liked'] = df['dish_liked'].fillna('').astype(str).apply(
            lambda x: [d.strip() for d in x.split(',') if d.strip()]
        )
        
        self._df = df
        print(f"Successfully loaded {len(self._df)} unique restaurant records into memory cache.")

    def get_all(self) -> List[Restaurant]:
        """Returns all canonical restaurant records."""
        records = self._df.to_dict(orient="records")
        return [Restaurant(**rec) for rec in records]

    def filter(self, criteria: FilterCriteria) -> List[Restaurant]:
        """Applies programmatic hierarchical filters to candidate restaurant records."""
        df = self._df.copy()
        
        # 1. Location (case-insensitive substring match with proxy routing for missing Bangalore areas)
        if criteria.location:
            loc_lower = criteria.location.lower()
            
            # Check if there is already a direct case-insensitive match in the dataset
            has_direct_match = df['location'].str.lower().str.contains(loc_lower, na=False).any()
            
            mapped_locations = None
            if not has_direct_match:
                # Map missing popular Bangalore areas to represented neighboring locations in the raw Zomato dataset
                location_routing = {
                    "hsr layout": ["hsr"],
                    "richmond town": ["residency road", "church street", "mg road", "lavelle road"],
                    "ulsoor": ["mg road", "church street", "indiranagar"],
                    "sadashivanagar": ["malleshwaram", "mg road"],
                    "hebbal": ["malleshwaram", "frazer town"],
                    "domlur": ["old airport road", "indiranagar", "koramangala 5th block"]
                }
                
                for key, val in location_routing.items():
                    if key in loc_lower or loc_lower in key:
                        mapped_locations = val
                        break
                        
            if mapped_locations:
                # Filter by any of the mapped locations
                masks = [df['location'].str.lower().str.contains(loc, na=False) for loc in mapped_locations]
                df = df[np.logical_or.reduce(masks)]
            else:
                df = df[df['location'].str.lower().str.contains(loc_lower, na=False)]
            
        # 2. Budget Band
        if criteria.budget:
            df = df[df['budget_band'] == criteria.budget]
            
        # 3. Cuisine (check cuisines column list, or fall back to representative liked dishes)
        if criteria.cuisine:
            cuisine_lower = criteria.cuisine.lower()
            
            # Map common cuisines to representative dishes in Zomato's dish_liked column
            cuisine_to_dishes = {
                "italian": ["pasta", "pizza", "lasagna", "lasagne", "risotto", "ravioli", "tiramisu", "bruschetta", "gnocchi", "panini", "spaghetti", "penne"],
                "mexican": ["taco", "nacho", "quesadilla", "burrito", "fajita", "guacamole", "enchilada", "churros"],
                "american": ["burger", "hot dog", "waffle", "steak", "fry", "fries", "ribs", "mac and cheese", "pancake"],
                "japanese": ["sushi", "ramen", "tempura", "sashimi", "yakitori", "udon", "gyoza"],
                "chinese": ["noodle", "momos", "momo", "dim sum", "manchurian", "chow mein", "spring roll"],
                "north indian": ["biryani", "roti", "naan", "paneer", "dal makhani", "butter chicken", "tikka", "kebab"],
                "south indian": ["dosa", "idli", "vada", "uttapam", "filter coffee", "sambar"]
            }
            
            dishes_for_cuisine = cuisine_to_dishes.get(cuisine_lower, [])
            
            def matches_cuisine(row):
                c_list = row.get('cuisines', [])
                if isinstance(c_list, (list, tuple, np.ndarray)):
                    if any(cuisine_lower in str(c).lower() for c in c_list):
                        return True
                
                d_list = row.get('dish_liked', [])
                if isinstance(d_list, (list, tuple, np.ndarray)):
                    if any(any(dish in str(d).lower() for dish in dishes_for_cuisine) for d in d_list):
                        return True
                return False
                
            mask = df.apply(matches_cuisine, axis=1)
            df = df.loc[mask]
            
        # 4. Min Rating
        if criteria.min_rating > 0:
            df = df[df['rating'] >= criteria.min_rating]
            
        records = df.to_dict(orient="records")
        return [Restaurant(**rec) for rec in records]
        
    def get_by_ids(self, ids: List[str]) -> List[Restaurant]:
        """Loads concrete Restaurant entities matching a set of ID strings."""
        df = self._df[self._df['id'].isin(ids)]
        records = df.to_dict(orient="records")
        return [Restaurant(**rec) for rec in records]
