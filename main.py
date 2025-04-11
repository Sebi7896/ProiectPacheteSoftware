import streamlit as st
import Utils

pagina_selectata = st.sidebar.radio("🔎 Alege o pagină:",
                                    ["Acasa",
                                     "Detalii Variabile",
                                     "Functii de grup",
                                     "Grafice",
                                     "GeoPandas",
                                     "Ratinguri",
                                     "Date lipsa",
                                     "Scalarea datelor",
                                     "Valori aberante",
                                     "Clusterizare"

                                     ])
Utils.script_sidebar(pagina_selectata)
