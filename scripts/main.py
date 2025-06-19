import sys
import os
import warnings

# BLOQUEO DE AVISOS Y MENSAJES DE LOGGING
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"
os.environ["STREAMLIT_LOGGING_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

# RUTA BASE RELATIVA DESDE EL ARCHIVO ACTUAL (SCRIPTS/MAIN.PY) HACIA SRC.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from streamlit_app import run_app

if __name__ == "__main__":
    run_app()

