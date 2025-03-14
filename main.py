import pandas as pd
import streamlit as st

import Utils

df = pd.read_csv('vgsales.csv', index_col=0)

pagina_selectata = st.sidebar.radio("🔎 Alege o pagină:", ["Acasa", "Detalii Variabile"])
Utils.script_sidebar(pagina_selectata, df)
