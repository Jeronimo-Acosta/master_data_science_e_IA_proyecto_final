# Skincare Product Recommender

Este repositorio contiene un sistema de recomendación para las necesidades de cada tipo de piel y compara productos de skincare europeos y coreanos, analizando sus ingredientes para sugerir alternativas similares entre ambas regiones y comparando precios. Proporciona link de compra para cada producto. 

## Estructura del repositorio

El repositorio consta de la siguiente estructura:
Proyecto Final/
├── data/
│ ├── jolse_products.csv (datos de productos coreanos, originalmente creado por scraping_jolse.ipynb)
│ └── promofarma_products.csv (datos de productos europeos, originalmente creado por scraping_promofarma.ipynb)
├── extras/
│ ├── infografia.png
│ └── elevator_pitch.docx
├── scripts/
│ ├── main.py (archivo principal)
│ ├── scraping_jolse.ipynb (scraper coreano)
│ └── scraping_promofarma.ipynb (scraper europeo)
└── src/
| ├── init.py
| ├── data_processing.py (limpieza y procesamiento de datos)
| ├── recommender.py (lógica de recomendación)
| └── streamlit_app.py (interfaz de usuario)

Aclaración: dado el tiempo que implica el scraping, y considerando que los sitios scrapeados cambian constantemente, la ejecución del main.py no gatilla los archivos de scraping. El scraping se ha hecho aparte y se han conservado los archivos csv generados para garantizar que no haya fallas a la hora de intentar reproducir el código.  

El main.py en scripts/ ejecuta la aplicación Streamlit (streamlit_app.py) que utiliza los módulos de procesamiento (data_processing.py) y recomendación (recommender.py).

## Pipeline

1. **Extracción de datos**
   - Los datasets originales se obtuvieron mediante web scraping. Contienen información de productos, incluyendo nombre, marca, ingredientes, precios, imagen y URLs de la tienda.

2. **Procesamiento**
   - Normalización de nombres de marcas (lista customizada de más de 500 marcas)
   - Limpieza de ingredientes (eliminación de valores nulos/vacíos)
   - Conversión de precios (won coreano a euros)

3. **Recomendación**
   - Sistema basado en reglas de ingredientes por tipo de piel/preocupaciones
   - Modelo de embeddings (`all-MiniLM-L6-v2`) para similitud semántica
   - Algoritmo híbrido (puntaje por ingredientes + similitud coseno)

4. **Interfaz**
   - Streamlit para selección de parámetros y visualización
   - Funcionalidad de comparación entre regiones

## Instalación

Para ejecutar este proyecto:

1. Clona el repositorio:
- git clone https://github.com/Jeronimo-Acosta/master_data_science_e_IA_proyecto_final
- cd scripts
- python -m venv venv (en Windows)
- .\venv\Scripts\actívate (en Windows)
- pip install -r requirements.txt
- streamlit run main.py