import streamlit as st
import Utils

pagina_selectata = st.sidebar.radio("🔎 Alege o pagină:", ["Acasa", "Detalii Variabile", "Date lipsa", "Functii de grup", "Grafice","Valori aberante","Scalarea datelor", "GeoPandas", "Ratinguri"])
Utils.script_sidebar(pagina_selectata)
