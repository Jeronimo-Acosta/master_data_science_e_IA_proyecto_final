from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SkincareRecommender:
    def __init__(self, df_eu, df_kr):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.df_eu = df_eu
        self.df_kr = df_kr
        self.ingredient_rules = {
            'skin_type': {
                'oily': ['niacinamide', 'salicylic acid', 'zinc', 'tea tree', 'clay', 'charcoal', 'witch hazel'],
                'dry': ['hyaluronic acid', 'glycerin', 'ceramide', 'squalane', 'shea butter', 'urea', 'panthenol'],
                'normal': ['antioxidants', 'peptides', 'green tea', 'vitamin e', 'ferulic acid'],
                'combination': ['niacinamide', 'hyaluronic acid', 'aloe vera', 'centella asiatica', 'propolis']
            },
            'conditions': {
                'rosacea': ['azelaic acid', 'centella asiatica', 'licorice root', 'green tea', 'aloe vera', 'chamomile'],
                'dermatitis': ['ceramide', 'oat', 'colloidal oatmeal', 'allantoin', 'panthenol', 'chamomile'],
                'psoriasis': ['salicylic acid', 'urea', 'coal tar', 'aloe vera', 'jojoba oil', 'vitamin d']
            },
            'concerns': {
                'acne': ['salicylic acid', 'benzoyl peroxide', 'niacinamide', 'retinol', 'tea tree', 'sulfur'],
                'calm': ['centella asiatica', 'aloe vera', 'chamomile', 'oat', 'green tea', 'licorice root'],
                'wrinkles': ['retinol', 'hyaluronic acid', 'peptides', 'vitamin c', 'niacinamide', 'bakuchiol'],
                'collagen': ['vitamin c', 'peptides', 'retinol', 'niacinamide', 'growth factors'],
                'pigmentation': ['vitamin c', 'niacinamide', 'arbutin', 'tranexamic acid', 'kojic acid', 'licorice root']
            }
        }
    
    def filter_products(self, df, skin_type, conditions, concerns):
        ingredients = (
            self.ingredient_rules['skin_type'].get(skin_type.lower(), []) +
            [i for c in conditions for i in self.ingredient_rules['conditions'].get(c.lower(), [])] +
            [i for c in concerns for i in self.ingredient_rules['concerns'].get(c.lower(), [])]
        )
        
        df['score'] = df['ingredients'].apply(
            lambda x: sum(ing.lower() in str(x).lower() for ing in ingredients)
        )
        return df[df['score'] > 0]
    
    def recommend(self, skin_type, conditions, concerns, region='eu', n=5):
        df = self.df_eu if region == 'eu' else self.df_kr
        filtered = self.filter_products(df.copy(), skin_type, conditions, concerns)
        
        if filtered.empty:
            return pd.DataFrame()
        
        query = f"Best product for {skin_type} skin with {', '.join(conditions)} to address {', '.join(concerns)}"
        product_texts = filtered.apply(
            lambda x: f"{x['brand'] if 'brand' in x else ''} {x['name']}: {x['ingredients']}", axis=1
        ).tolist()
        
        filtered['similarity'] = cosine_similarity(
            self.model.encode([query]),
            self.model.encode(product_texts)
        )[0]
        
        return filtered.sort_values(['similarity', 'score'], ascending=False).head(n)

    def compare_products(self, selected_row, region, compare_to, n=5):
        product = selected_row.copy()
        other_df = self.df_kr if compare_to == 'kr' else self.df_eu
        
        product_ingredients = set(str(product['ingredients']).lower().split(','))
        other_df['common_ingredients'] = other_df['ingredients'].apply(
            lambda x: len(product_ingredients.intersection(set(str(x).lower().split(',')))))
        
        query = f"{product.get('brand', '')} {product['name']}: {product['ingredients']}"
        other_texts = other_df.apply(
            lambda x: f"{x.get('brand', '')} {x['name']}: {x['ingredients']}", 
            axis=1
        ).tolist()
        
        other_df['similarity'] = cosine_similarity(
            self.model.encode([query]),
            self.model.encode(other_texts)
        )[0]
        
        max_common = other_df['common_ingredients'].max() or 1
        max_sim = other_df['similarity'].max() or 1
        other_df['match_score'] = (
            0.6 * (other_df['common_ingredients'] / max_common) + 
            0.4 * (other_df['similarity'] / max_sim))
        
        return product, other_df.sort_values(['match_score', 'common_ingredients'], ascending=False).head(n)