import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER'] = 'false'

# Set page config FIRST
st.set_page_config(page_title="Skincare Recommender", page_icon="🌸", layout="wide")

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Data Loading and Preprocessing
@st.cache_data
def load_data():
    df_kr = pd.read_csv('jolse_products.csv')
    df_eu = pd.read_csv('promofarma_products.csv')
    
    # EU data cleaning
    df_eu.columns = [col.lower() for col in df_eu.columns]
    renombres = {
        'nombre': 'name', 'marca': 'brand', 'precio': 'price',
        'ingredientes': 'ingredients', 'url_producto': 'product_url',
        'imagen_url': 'image_url'
    }
    df_eu = df_eu.rename(columns=renombres)
    
    # Handle missing/empty ingredients
    for df in [df_kr, df_eu]:
        df.dropna(subset=['ingredients'], inplace=True)
        df = df[df['ingredients'].str.strip().str.lower() != 'no encontrado']
        df = df[df['ingredients'].str.strip() != '']

    if 'brand' in df_kr.columns:
        # Drop rows where brand is "The Ordinary" (case insensitive)
        df_kr = df_kr[~df_kr['brand'].str.lower().eq('the ordinary')]
    
    # KR price conversion
    if 'price' in df_kr.columns:
        df_kr['price'] = (
            df_kr['price'].astype(str)
            .str.replace(r'[^0-9.,]', '', regex=True)
            .str.replace(',', '.')
            .astype(float) * 0.88
        ).round(2)
    
    return df_kr, df_eu

df_kr, df_eu = load_data()

# 2. Brand Extraction (EU only)
brands_list = [
    "Avène", "Avène", "Avène", "Avène", "Endocare", "Avène", "Rosacure", "Endocare", "Xhekpon", "ISDIN",
    "La Roche-Posay", "Bioderma", "Endocare", "Retincare", "La Roche Posay", "Avène", "Biretix", "Bioderma",
    "Biretix", "CeraVe", "Biretix", "Rosacure", "Bioderma", "Rosacure", "Boderm", "SVR", "La Roche-Posay",
    "La Roche-Posay", "CeraVe", "Avène", "SVR", "Endocare", "Avène", "Rosacure", "Martiderm", "Avène",
    "Endocare", "CeraVe", "Boderm", "Skin Resist", "Acm", "Neoretin", "Avène", "Biretix", "Biretix",
    "Skin Resist", "La Roche-Posay", "Uriage Roséliane", "Avène", "Sesderma", "Bioderma", "Avène", "Galenicum",
    "Endocare", "Martiderm", "Prisma", "Avène", "Avène", "Bioderma", "Cetaphil", "Rilastil", "La Roche-Posay",
    "Avène", "Endocare", "Avène", "Avène", "Avène", "Avène", "Endocare", "La Roche-Posay", "Erborian",
    "Bioderma", "Avène", "Avène", "SVR", "Bioderma", "La Roche Posay", "Erborian", "SVR", "La Roche-Posay",
    "Bioderma", "Rosacure", "Bioderma", "Noviderm", "Rilastil", "ISDIN", "Endocare", "La Roche-Posay",
    "Avène", "Bioderma", "LetiSR", "Endocare", "Svr", "Endocare", "SVR", "Avène", "Bioderma", "Biretix",
    "Avène", "Uriage", "Sesderma", "Bella Aurora", "Lovren", "Biretix", "Uriage", "Endocare", "Medik8",
    "Galénic", "ISDIN", "Neoretin", "Avène", "Boderm", "Neoretin", "LetiSR", "La Roche-Posay", "Avène",
    "Erborian", "Ozoaqua", "Cetaphil", "SVR", "La Roche-Posay", "ISDIN", "Avène", "Bioderma", "Eucerin",
    "Avène", "Avène", "Avène", "Avène", "Martiderm", "Avène", "Neoretin", "Sesderma", "Nuxe", "Avène",
    "Avène", "Bioderma", "Endocare", "Uriage", "ISDIN", "Isispharma", "Eucerin", "Martiderm", "Endocare",
    "Avène", "Heliocare", "Eucerin", "Avène", "Avène", "Skin Resist", "Bioderma", "Erborian", "Erborian",
    "CeraVe", "Avène", "Repavar", "La Roche-Posay", "La Roche-Posay", "Tensoderm", "Avène", "Gh",
    "La Roche-Posay", "Sesderma", "CeraVe", "Eucerin", "Nuxe", "SVR", "SVR", "Ozoaqua", "SVR", "Sesderma",
    "Martiderm", "Avène", "Medik8", "Sesderma", "La Roche-Posay", "Martiderm", "Nuxe", "Avène", "Compeed",
    "Sesderma", "Bioderma", "Comodynes", "Bella Aurora", "Isdinceutics", "Vital Plus JAL", "La Roche Posay",
    "Avène", "Avène", "Avène", "Avène", "Avène", "ISDIN", "Rosaderm", "Institut Esthederm", "Repavar",
    "Dodot", "La Roche-Posay", "La Roche Posay", "Farline", "Bella Aurora", "Nuxe", "Bioderma", "Bioderma",
    "Erborian", "SVR", "Apivita", "Cetaphil", "Neoretin", "SVR", "Bioderma", "Babaria", "ISDIN", "Letisr",
    "NeoStrata", "Germinal", "Armonia", "ISDIN", "Sesderma", "Avène", "Xemose", "CeraVe", "Uriage", "Uriage",
    "Singuladerm", "La Roche-Posay", "La Roche-Posay", "Avène", "Endocare", "La Roche-Posay", "Bioderma",
    "Primaderm", "Ducray", "Uriage", "Bioderma", "SVR", "SVR", "Uriage", "Avène", "La Roche-Posay", "Avène",
    "NeoStrata", "Endocare", "Sesderma", "Rosaderm", "Skin Perfection", "Gh", "Patyka", "Cerave", "Avène",
    "Compeed", "Avène", "Avène", "NeoStrata", "Eucerin", "La Roche-Posay", "Freshly Cosmetics", "Sesderma",
    "Avène", "Eucerin", "SVR", "LetiSR", "Avène", "Martiderm", "Endocare", "Endocare", "Avene", "M Fhaktor",
    "Ducray", "Uriage", "ISDIN", "Herbora", "Sesderma", "SVR", "A-Derma", "ISDIN", "Bioderma",
    "Freshly Cosmetics", "Sensilis", "Vital Plus", "Postquam", "Uriage", "La Roche-Posay", "Eucerin",
    "Embryolisse", "Embryolisse", "Be+", "Singuladerm", "Sesderma", "Medik8", "Gh", "GH", "GH", "Martiderm",
    "Eucerin", "Avène", "Erborian", "Vichy", "Martiderm", "Avène", "Lierac", "Compeed", "Uriage", "Eucerin",
    "Aposán", "Avène", "Avène", "Avène", "Avène", "Avène", "Avene", "Eucerin", "SVR", "Bella Aurora",
    "Endocare", "Leti AT4", "Uriage", "ISDIN", "Galenicum", "La Roche-Posay", "Rilastil", "Neoretin",
    "Avène", "Postquam", "A-Derma", "SVR", "Cetaphil", "Biotherm", "NAN", "Neostrata", "Eucerin", "Vichy",
    "Uriage", "Bioderma", "Vichy", "Endocare", "Sesderma", "Tensoderm", "Sesderma", "Repavar", "Uriage",
    "Cosmeclinik", "Avène", "Medik8", "Avène", "Gh", "GH", "Primaderm", "Reguven", "Bromatech", "ISDIN",
    "Arturo", "Gh", "GH", "La Roche-Posay", "Armonía", "Nuxe", "SingulaDerm", "Sesderma", "Bella Aurora",
    "Biretix", "Vichy", "Lovren", "Avène", "Avène", "Avène", "Avène", "Bromatech", "Neoretin", "SVR",
    "Vichy", "Ducray", "Nuxe", "Martiderm", "AOKlabs", "Galenicum", "ISDIN", "Erborian", "Cetaphil",
    "Postquam", "Vichy", "Vichy", "La Roche-Posay", "Uresim", "Avène", "La Roche-Posay", "Lutsine",
    "Eucerin", "Vichy", "Nuxe", "Avène", "Avène", "Bioderma", "Cetaphil", "Sesderma", "Ducray", "Sesderma",
    "Medik8", "Filorga", "Nivea", "Weleda", "Evea", "Martiderm", "Cattier", "Jowae", "Martiderm", "Nuxe",
    "Uriage", "Nuxe", "Nuxe", "SVR", "CosmeClinik", "Vichy", "Neutrogena", "Bioderma", "Avène", "Filorga",
    "Cetaphil", "SVR", "Avène", "Avène", "Avène", "Avène", "Bioderma", "Endocare", "SVR", "Naobay", "Vichy",
    "Neutrogena", "A-Derma", "A-Derma", "Vichy", "Sensilis", "Bioderma", "Avène", "SingulaDerm",
    "La Roche-Posay", "Unique", "Lashile", "Vichy", "Eucerin", "ISDIN", "Sesderma", "Isdin", "Isdin",
    "Sesderma", "Sensilis", "Freshly Cosmetics", "Freshly Cosmetics", "Endocare", "ROC", "Rilastil", "Babé",
    "Galénic", "Avène", "Eucerin", "Neutrogena", "SVR", "Sesderma", "La Roche-Posay", "ISDIN", "Martiderm",
    "Martiderm", "La Roche-Posay", "Eucerin", "Jowaé", "Ducrem", "A-Derma", "Avène", "Uriage", "ISDIN",
    "Endocare", "Sesderma", "Bioderma", "Bioderma", "Uresim", "Sesderma", "Galénic", "Sato", "Avène",
    "Avène", "Avène", "Avène", "Be+", "ISDIN", "NeoStrata", "Neutrogena", "Medik8", "Darphin", "ISDIN",
    "CeraVe", "Cetaphil", "Martiderm", "GH", "CeraVe", "Emolienta", "Repavar", "Arturo", "Be+",
    "NeoStrata", "Eucerin", "Arturo Alba", "Arturo Alba", "La Roche-Posay", "ISDIN", "ISDIN", "Licotriz",
    "Sesderma", "Martiderm", "SVR", "La Roche-Posay", "Crema", "Gh", "SVR", "CeraVe", "ISDIN", "Lierac",
    "Avène", "Lovren", "Avène", "Cattier", "Nuxe", "ACM", "Sensilis", "Martiderm", "Weleda", "Svr",
    "Embryolisse", "SVR", "AOKlabs", "Farline", "Vichy", "Bioderma", "Apivita", "Avène", "Avène", "Avène",
    "Avène", "Clinicalfarma", "Vichy", "Eucerin", "Endocare", "Uriage", "Uriage", "Avene", "Martiderm",
    "Eucerin", "Exomega", "Apivita", "Institut Esthederm", "A-Derma", "A-Derma", "Bioderma", "Unique",
    "Primaderm", "Bioderma", "Sensilis", "SVR", "Sesderma", "Vichy", "SVR", "Salustar", "Singuladerm",
    "ISDIN", "Dr Fillermast", "Vichy", "Vichy", "Sensilis", "Avène", "Eucerin", "Bioderma", "ISDIN",
    "Sesderma", "Freshly Cosmetics", "Germinal", "Segle", "Rilastil", "Uriage", "Biretix", "SVR", "Filorga",
    "NeoStrata", "Darphin", "Filorga", "CeraVe", "NeoStrata", "Santiveri", "Naobay", "Nuxe", "Nuxe",
    "Eucerin", "LetiAT4", "La Roche-Posay", "Avène", "Sensilis"
]

def extract_brand(nombre):
    for marca in sorted(set(brands_list), key=lambda x: -len(x)):
        if nombre.lower().startswith(marca.lower()):
            return pd.Series([marca, nombre[len(marca):].strip(" ,.-")])
    return pd.Series(["", nombre])

df_eu[['brand', 'name']] = df_eu['name'].apply(extract_brand)
df_eu['brand'] = df_eu['brand'].str.upper()

# 3. Recommendation Engine (keep your existing class)
class SkincareRecommender:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
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
        df = df_eu if region == 'eu' else df_kr
        filtered = self.filter_products(df.copy(), skin_type, conditions, concerns)
        
        if filtered.empty:
            return pd.DataFrame()
        
        # Semantic ranking
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
        """Compare a selected product to ALL alternatives from another region."""
        product = selected_row.copy()
        other_df = df_kr if compare_to == 'kr' else df_eu
        
        # Calculate ingredient matches with ALL products
        product_ingredients = set(str(product['ingredients']).lower().split(','))
        other_df['common_ingredients'] = other_df['ingredients'].apply(
            lambda x: len(product_ingredients.intersection(set(str(x).lower().split(',')))))
        
        # Calculate semantic similarity with ALL products
        query = f"{product.get('brand', '')} {product['name']}: {product['ingredients']}"
        other_texts = other_df.apply(
            lambda x: f"{x.get('brand', '')} {x['name']}: {x['ingredients']}", 
            axis=1
        ).tolist()
        
        other_df['similarity'] = cosine_similarity(
            self.model.encode([query]),
            self.model.encode(other_texts)
        )[0]
        
        # Normalize and combine scores (60% ingredients, 40% similarity)
        max_common = other_df['common_ingredients'].max() or 1
        max_sim = other_df['similarity'].max() or 1
        other_df['match_score'] = (
            0.6 * (other_df['common_ingredients'] / max_common) + 
            0.4 * (other_df['similarity'] / max_sim))
        
        return product, other_df.sort_values(['match_score', 'common_ingredients'], ascending=False).head(n)


def display_comparison(original, matches, original_region):
    st.subheader(f"🔍 Original {original_region.upper()} Product:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {original['name']}")
        if 'brand' in original:
            st.markdown(f"**Brand:** {original['brand']}")
        st.markdown(f"**Price:** {original.get('price', 'N/A')}")
    with col2:
        if 'image_url' in original and pd.notna(original['image_url']):
            st.image(original['image_url'], width=150)
    
    st.markdown(f"**Ingredients:** {original['ingredients'][:200]}...")
    
    st.subheader(f"🌸 Best {('kr' if original_region == 'eu' else 'eu').upper()} Alternatives:")
    
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        with st.expander(f"Option {idx}: {row.get('brand', '')} {row['name']}"):
            cols = st.columns([1, 3])
            with cols[0]:
                if 'image_url' in row and pd.notna(row['image_url']):
                    st.image(row['image_url'], width=150)
            
            with cols[1]:
                if 'brand' in row:
                    st.markdown(f"**Brand:** {row['brand']}")
                st.markdown(f"**Name:** {row['name']}")
                
                # Price comparison
                price_diff = ""
                try:
                    original_price = float(str(original.get('price', '0')).replace('€', '').replace(',', '.').strip())
                    row_price = float(str(row.get('price', '0')).replace('€', '').replace(',', '.').strip())
                    
                    if row_price < original_price:
                        price_diff = f" (€{original_price-row_price:.2f} cheaper)"
                    elif row_price > original_price:
                        price_diff = f" (€{row_price-original_price:.2f} more expensive)"
                    else:
                        price_diff = " (same price)"
                except:
                    price_diff = " (price comparison unavailable)"
                    
                st.markdown(f"**Price:** {row.get('price', 'N/A')}{price_diff}")
                st.markdown(f"**Match Score:** {row['match_score']:.2f} (Ingredients: {row['common_ingredients']}, Similarity: {row['similarity']:.2f})")
                
                if 'product_url' in row and pd.notna(row['product_url']):
                    st.markdown(f"[Product Link]({row['product_url']})")
                
                # Display first 5 ingredients
                ingredients = [i.strip() for i in str(row['ingredients']).lower().split(',')[:5]]
                st.markdown(f"**Key Ingredients:** {', '.join(ingredients)}...")






def reset_session_state():
    """Reset all session state variables to their initial values"""
    st.session_state.compare_mode = False
    st.session_state.selected_product = None
    st.session_state.original_region = None
    st.session_state.show_recommendations = False
    # Reset form fields by clearing their session state values
    if 'skin_type_select' in st.session_state:
        del st.session_state.skin_type_select
    if 'conditions_select' in st.session_state:
        del st.session_state.conditions_select
    if 'concerns_select' in st.session_state:
        del st.session_state.concerns_select

def main():
    # Initialize session state
    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'original_region' not in st.session_state:
        st.session_state.original_region = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # Custom CSS
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #ff6b6b;
            color: white;
        }
        .stSelectbox, .stTextInput {
            margin-bottom: 10px;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
        }
        .reset-button {
            background-color: #6b6bff !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌸 Skincare Product Recommender")
    st.markdown("Find the perfect skincare products for your needs and compare EU and Korean alternatives.")
    
    recommender = SkincareRecommender()
    
    # User inputs in sidebar
    with st.sidebar:
        st.header("Your Skin Profile")
        skin_type = st.selectbox(
            "Skin type",
            ["Oily", "Dry", "Normal", "Combination"],
            index=2,  # Default to "Normal"
            key="skin_type_select"
        ).lower()
        
        conditions = st.multiselect(
            "Skin conditions (select all that apply)",
            ["Rosacea", "Dermatitis", "Psoriasis"],
            key="conditions_select"
        )
        
        concerns = st.multiselect(
            "Primary concerns (select all that apply)",
            ["Acne", "Calm", "Wrinkles", "Collagen", "Pigmentation"],
            key="concerns_select"
        )
        
        st.markdown("---")
        
        # Get Recommendations button
        if st.button("Get Recommendations"):
            st.session_state.show_recommendations = True
            st.session_state.compare_mode = False
            st.rerun()
        
        # Reset button - now properly clears all fields
        if st.button("🔄 Start from scratch", type="primary", key="reset_button"):
            reset_session_state()
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Note:** This app recommends products based on ingredient analysis and semantic similarity.")




def reset_session_state():
    """Reset all session state variables to their initial values"""
    st.session_state.compare_mode = False
    st.session_state.selected_product = None
    st.session_state.original_region = None
    st.session_state.show_recommendations = False
    # Clear the form fields by removing their session state values
    if 'skin_type_select' in st.session_state:
        del st.session_state.skin_type_select
    if 'conditions_select' in st.session_state:
        del st.session_state.conditions_select
    if 'concerns_select' in st.session_state:
        del st.session_state.concerns_select

def main():
    # Initialize session state
    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'original_region' not in st.session_state:
        st.session_state.original_region = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # Custom CSS
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #ff6b6b;
            color: white;
        }
        .stSelectbox, .stTextInput {
            margin-bottom: 10px;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
        }
        .reset-button {
            background-color: #6b6bff !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌸 Skincare Product Recommender")
    st.markdown("Find the perfect skincare products for your needs and compare EU and Korean alternatives.")
    
    recommender = SkincareRecommender()
    
    # User inputs in sidebar
    with st.sidebar:
        st.header("Your Skin Profile")
        
        # Set default values based on session state or None
        skin_type_default = st.session_state.get('skin_type_select', "Normal")
        conditions_default = st.session_state.get('conditions_select', [])
        concerns_default = st.session_state.get('concerns_select', [])
        
        skin_type = st.selectbox(
            "Skin type",
            ["Oily", "Dry", "Normal", "Combination"],
            index=["Oily", "Dry", "Normal", "Combination"].index(skin_type_default) if skin_type_default in ["Oily", "Dry", "Normal", "Combination"] else 2,
            key="skin_type_select"
        ).lower()
        
        conditions = st.multiselect(
            "Skin conditions (select all that apply)",
            ["Rosacea", "Dermatitis", "Psoriasis"],
            default=conditions_default,
            key="conditions_select"
        )
        
        concerns = st.multiselect(
            "Primary concerns (select all that apply)",
            ["Acne", "Calm", "Wrinkles", "Collagen", "Pigmentation"],
            default=concerns_default,
            key="concerns_select"
        )
        
        st.markdown("---")
        
        # Get Recommendations button
        if st.button("Get Recommendations"):
            st.session_state.show_recommendations = True
            st.session_state.compare_mode = False
            st.rerun()
        
        # Reset button - now properly clears all fields
        if st.button("🔄 Start from scratch", type="primary", key="reset_button"):
            reset_session_state()
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Note:** This app recommends products based on ingredient analysis and semantic similarity.")

    if st.session_state.compare_mode:
        # Comparison view
        original = st.session_state.selected_product
        region = st.session_state.original_region
        compare_to = 'kr' if region == 'eu' else 'eu'
        
        with st.spinner("Finding comparable products..."):
            original, matches = recommender.compare_products(
                original, 
                region, 
                compare_to                
            )
            display_comparison(original, matches, region)
        
        if st.button("← Back to recommendations"):
            st.session_state.compare_mode = False
            st.rerun()
            
    elif st.session_state.show_recommendations:
        # Recommendations view
        with st.spinner("Finding the best products for you..."):
            # Display EU recommendations
            st.header("🇪🇺 European Products")
            eu_results = recommender.recommend(skin_type, conditions, concerns, 'eu')
            
            if eu_results is not None and not eu_results.empty:
                for idx, row in eu_results.iterrows():
                    with st.expander(f"{row.get('brand', '')} {row['name']} - €{row.get('price', 'N/A')}"):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            if 'image_url' in row and pd.notna(row['image_url']):
                                st.image(row['image_url'], width=150)
                        
                        with cols[1]:
                            if 'brand' in row:
                                st.markdown(f"**Brand:** {row['brand']}")
                            st.markdown(f"**Name:** {row['name']}")
                            st.markdown(f"**Price:** {row.get('price', 'N/A')}")
                            if 'product_url' in row and pd.notna(row['product_url']):
                                st.markdown(f"[Product Link]({row['product_url']})")
                            
                            # Button for comparison
                            if st.button(f"Compare with Korean alternatives", key=f"eu_compare_{idx}"):
                                st.session_state.compare_mode = True
                                st.session_state.selected_product = eu_results.loc[idx]
                                st.session_state.original_region = 'eu'
                                st.rerun()
                
            else:
                st.warning("No European products found matching your criteria")
            
            # Display KR recommendations
            st.header("🇰🇷 Korean Products")
            kr_results = recommender.recommend(skin_type, conditions, concerns, 'kr')
            
            if kr_results is not None and not kr_results.empty:
                for idx, row in kr_results.iterrows():
                    with st.expander(f"{row.get('brand', '')} {row['name']} - €{row.get('price', 'N/A')}"):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            if 'image_url' in row and pd.notna(row['image_url']):
                                st.image(row['image_url'], width=150)
                        
                        with cols[1]:
                            if 'brand' in row:
                                st.markdown(f"**Brand:** {row['brand']}")
                            st.markdown(f"**Name:** {row['name']}")
                            st.markdown(f"**Price:** {row.get('price', 'N/A')}")
                            if 'product_url' in row and pd.notna(row['product_url']):
                                st.markdown(f"[Product Link]({row['product_url']})")
                            
                            # Button for comparison
                            if st.button(f"Compare with European alternatives", key=f"kr_compare_{idx}"):
                                st.session_state.compare_mode = True
                                st.session_state.selected_product = kr_results.loc[idx]
                                st.session_state.original_region = 'kr'
                                st.rerun()
            else:
                st.warning("No Korean products found matching your criteria")

if __name__ == "__main__":
    main()