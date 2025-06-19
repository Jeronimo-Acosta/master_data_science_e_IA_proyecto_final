import pandas as pd
import numpy as np


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


def load_data():
    """    
    Carga y preprocesa los datos de productos de skincare desde archivos CSV. Devuelve dos DataFrames 
    con datos de productos coreanos y europeos. Lee archivos CSV del directorio data, estandariza 
    nombres de columnas para datos EU, maneja ingredientes faltantes/vacíos, filtra la marca 
    'The Ordinary' de los datos coreanos (no es coreana), y convierte precios coreanos de KRW a EUR 
    (tasa de conversión 0.88)
    """
    df_kr = pd.read_csv('../data/jolse_products.csv')
    df_eu = pd.read_csv('../data/promofarma_products.csv')
    
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


def extract_brand(nombre):
    """
    Extrae el nombre de la marca del nombre del producto usando una lista predefinida de marcas (el
    scraping no llegó a hacerlo por sí sólo, y fue muy difícil encontrar otras páginas scrapeables con
    la información necesaria para el proyecto. Se le pasó la lista entera a ChatGPT para que
    automáticamente se genere la lista a extraer ).
    """
    for marca in sorted(set(brands_list), key=lambda x: -len(x)):
        if nombre.lower().startswith(marca.lower()):
            return pd.Series([marca, nombre[len(marca):].strip(" ,.-")])
    return pd.Series(["", nombre])


def process_data(df_eu):
    """
    Procesa los datos europeos extrayendo marcas de los nombres de productos. Entra df_eu,
    devuelve el mismo DataFrame con columnas adicionales 'brand' y 'name' procesadas. Añade columna 
    'brand' con la marca extraída, modifica columna 'name' eliminando la marca y convierte marcas
    a mayúsculas
    """
    df_eu[['brand', 'name']] = df_eu['name'].apply(extract_brand)
    df_eu['brand'] = df_eu['brand'].str.upper()
    return df_eu