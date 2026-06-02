import pandas as pd

class SchemaNormalizer:
    def __init__(self, budget_low_max: float = 500.0, budget_medium_max: float = 1500.0):
        self.budget_low_max = budget_low_max
        self.budget_medium_max = budget_medium_max

    def _clean_rate(self, rate_val) -> float:
        """Converts strings like '4.1/5' or 'NEW' to a float rating."""
        if pd.isna(rate_val):
            return 0.0
        
        rate_str = str(rate_val).strip()
        if rate_str in ('NEW', '-', ''):
            return 0.0
            
        try:
            if '/' in rate_str:
                return float(rate_str.split('/')[0].strip())
            return float(rate_str)
        except Exception:
            return 0.0

    def _clean_cost(self, cost_val) -> float:
        """Converts cost strings like '800' or '1,200' to float."""
        if pd.isna(cost_val):
            return 0.0
        
        cost_str = str(cost_val).replace(',', '').strip()
        try:
            return float(cost_str)
        except ValueError:
            return 0.0

    def _assign_budget_band(self, cost: float) -> str:
        """Maps continuous cost into discrete budget bands for filtering."""
        if cost <= 0.0:
            return "unknown"
        if cost <= self.budget_low_max:
            return "low"
        elif cost <= self.budget_medium_max:
            return "medium"
        else:
            return "high"

    def _normalize_location(self, loc_val) -> str:
        """Collapses specific blocks, sectors, or spelling variations to canonical parent locations."""
        if pd.isna(loc_val):
            return "Unknown"
        loc = str(loc_val).strip()
        loc_lower = loc.lower()
        
        if "koramangala" in loc_lower:
            return "Koramangala"
        if "hsr" in loc_lower:
            return "HSR Layout"
        if "jayanagar" in loc_lower:
            return "Jayanagar"
        if "jp nagar" in loc_lower:
            return "JP Nagar"
        if "whitefield" in loc_lower:
            return "Whitefield"
        if "btm" in loc_lower:
            return "BTM"
        if "indiranagar" in loc_lower or "indira nagar" in loc_lower:
            return "Indiranagar"
        if "malleshwaram" in loc_lower or "malleswaram" in loc_lower:
            return "Malleshwaram"
        if "banashankari" in loc_lower:
            return "Banashankari"
        if "electronic city" in loc_lower:
            return "Electronic City"
        if "sarjapur" in loc_lower:
            return "Sarjapur Road"
        if "kalyan nagar" in loc_lower:
            return "Kalyan Nagar"
        if "kammanahalli" in loc_lower:
            return "Kammanahalli"
        if "basavanagudi" in loc_lower:
            return "Basavanagudi"
        if "richmond" in loc_lower:
            return "Richmond Town"
        if "sadashiva" in loc_lower or "sadashivanagar" in loc_lower:
            return "Sadashivanagar"
        if "bannerghatta" in loc_lower:
            return "Bannerghatta Road"
        if "old airport road" in loc_lower:
            return "Old Airport Road"
        if "marathahalli" in loc_lower or "marathalli" in loc_lower:
            return "Marathahalli"
        if "bellandur" in loc_lower:
            return "Bellandur"
        if "hebbal" in loc_lower:
            return "Hebbal"
        if "ulsoor" in loc_lower:
            return "Ulsoor"
        if "domlur" in loc_lower:
            return "Domlur"
            
        return loc

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Normalizing dataset schema...")
        
        # 1. Assign unique IDs
        df['id'] = [f"res_{i}" for i in range(len(df))]
        
        # 2. Clean Name
        df['name'] = df['name'].fillna('Unknown').astype(str).str.strip()
        
        # 3. Clean Location
        raw_location = df['listed_in(city)'].fillna(df['location']).fillna('Unknown')
        df['location'] = raw_location.apply(self._normalize_location)
        
        # 4. Clean Cuisines
        df['cuisines'] = df['cuisines'].fillna('').astype(str).apply(
            lambda x: [c.strip() for c in x.split(',') if c.strip()]
        )
        
        # 5. Clean Rating
        df['rating'] = df['rate'].apply(self._clean_rate)
        
        # 6. Clean Cost
        df['estimated_cost'] = df['approx_cost(for two people)'].apply(self._clean_cost)
        
        # 7. Assign Budget Band
        df['budget_band'] = df['estimated_cost'].apply(self._assign_budget_band)
        
        # 8. Clean Votes
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
        
        # 8b. Clean dish_liked
        df['dish_liked'] = df['dish_liked'].fillna('').astype(str).apply(
            lambda x: [d.strip() for d in x.split(',') if d.strip()]
        )

        # 9. Deduplication (dedupe based on name and location, keeping the highest rated entry)
        # Deduplicating here allows us to have a smaller, clean candidate base.
        before_drop = len(df)
        df_sorted = df.sort_values('rating', ascending=False)
        df_deduped = df_sorted.drop_duplicates(subset=['name', 'location']).copy()
        dropped = before_drop - len(df_deduped)
        print(f"Dropped {dropped} duplicate rows.")

        # 10. Generate text representation for local vector embedding
        def make_text_rep(row):
            cuisines_str = ", ".join(row['cuisines']) if row['cuisines'] else "none"
            dishes_str = ", ".join(row['dish_liked']) if row['dish_liked'] else "none"
            return (
                f"Name: {row['name']} | "
                f"Location: {row['location']} | "
                f"Cuisines: {cuisines_str} | "
                f"Rating: {row['rating']} | "
                f"Cost for two: {row['estimated_cost']} | "
                f"Dishes Liked: {dishes_str}"
            )
            
        df_deduped['text_representation'] = df_deduped.apply(make_text_rep, axis=1)

        # 11. Extract Canonical Domain Model Fields
        canonical_columns = [
            'id', 'name', 'location', 'cuisines', 'rating', 'estimated_cost', 'budget_band', 'votes', 'dish_liked', 'text_representation'
        ]
        df_normalized = df_deduped[canonical_columns].copy()
        
        print(f"Normalized {len(df_normalized)} unique canonical records.")
        return df_normalized
