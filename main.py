import pandas as pd
import streamlit as st

import Utils

pagina_selectata = st.sidebar.radio("🔎 Alege o pagină:", ["Acasa", "Detalii Variabile", "Date lipsa"])
Utils.script_sidebar(pagina_selectata)
