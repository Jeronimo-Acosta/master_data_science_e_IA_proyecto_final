import streamlit as st
import pandas as pd
import warnings
import os
from data_processing import load_data, process_data
from recommender import SkincareRecommender

# Configuración inicial
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER'] = 'false'
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

def reset_session_state():
    st.session_state.compare_mode = False
    st.session_state.selected_product = None
    st.session_state.original_region = None
    st.session_state.show_recommendations = False
    if 'skin_type_select' in st.session_state:
        del st.session_state.skin_type_select
    if 'conditions_select' in st.session_state:
        del st.session_state.conditions_select
    if 'concerns_select' in st.session_state:
        del st.session_state.concerns_select

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
                
                ingredients = [i.strip() for i in str(row['ingredients']).lower().split(',')[:5]]
                st.markdown(f"**Key Ingredients:** {', '.join(ingredients)}...")

def run_app():
    # Inicialización de datos y modelo
    df_kr, df_eu = load_data()
    df_eu = process_data(df_eu)
    recommender = SkincareRecommender(df_eu, df_kr)
    
    # Inicializar estado de sesión
    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'original_region' not in st.session_state:
        st.session_state.original_region = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # Configuración de la página
    st.set_page_config(page_title="Skincare Recommender", page_icon="🌸", layout="wide")
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
    
    # Sidebar
    with st.sidebar:
        st.header("Your Skin Profile")
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
        
        if st.button("Get Recommendations"):
            st.session_state.show_recommendations = True
            st.session_state.compare_mode = False
            st.rerun()
        
        if st.button("🔄 Start from scratch", type="primary", key="reset_button"):
            reset_session_state()
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Note:** This app recommends products based on ingredient analysis and semantic similarity.")

    # Lógica principal de la aplicación
    if st.session_state.compare_mode:
        original = st.session_state.selected_product
        region = st.session_state.original_region
        compare_to = 'kr' if region == 'eu' else 'eu'
        
        with st.spinner("Finding comparable products..."):
            original, matches = recommender.compare_products(original, region, compare_to)
            display_comparison(original, matches, region)
        
        if st.button("← Back to recommendations"):
            st.session_state.compare_mode = False
            st.rerun()
            
    elif st.session_state.show_recommendations:
        with st.spinner("Finding the best products for you..."):
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
                            
                            if st.button(f"Compare with Korean alternatives", key=f"eu_compare_{idx}"):
                                st.session_state.compare_mode = True
                                st.session_state.selected_product = eu_results.loc[idx]
                                st.session_state.original_region = 'eu'
                                st.rerun()
            else:
                st.warning("No European products found matching your criteria")
            
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
                            
                            if st.button(f"Compare with European alternatives", key=f"kr_compare_{idx}"):
                                st.session_state.compare_mode = True
                                st.session_state.selected_product = kr_results.loc[idx]
                                st.session_state.original_region = 'kr'
                                st.rerun()
            else:
                st.warning("No Korean products found matching your criteria")