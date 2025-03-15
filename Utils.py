import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vgsales.csv', index_col=0)
st.session_state['filtered_df'] = df.copy()
filter_dict = {x: "" for x in df.columns}


def analiza_joc(nume, descriere):
    st.subheader(nume)
    st.markdown(descriere)


def script_sidebar(pagina_selectata):
    if pagina_selectata == "Acasa":
        st.header("🏠 Acasa")
        st.subheader("Dataframe-ul")
        col1, col2 = st.columns([1, 2])  # Adjust the ratio for better layout

        with col1:
            for column in st.session_state['filtered_df'].columns:
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    st.text_input(label=column, max_chars=20, key=column, on_change=filtrare, args=[column])
                    st.selectbox(label=column, key=column + ' box', options=['Equal', 'Bigger', 'Smaller'],
                                 label_visibility='collapsed',on_change=filtrare(column))
                else:
                    st.text_input(label=column, max_chars=20, key=column, args=[column],on_change=filtrare)
            st.button(label="Resetare campuri", on_click=resetare)
        with col2:
            st.dataframe(st.session_state['filtered_df'], height=1400)

    if pagina_selectata == "Detalii Variabile":
        st.header("Detalii Variabile")
        date_joc = {
            "🏆 Rank-ul": "Acesta reprezintă poziția jocului în clasamentul global al vânzărilor.",
            "🎮 Name": "Denumirea oficială a jocului.",
            "🕹️ Platform": "Prima platformă pe care a fost lansat jocul **Wii,PS3,PC**.",
            "📅 Year": "Anul lansării jocului",
            "📌 Genre": "Genul jocului **Action,Racing,Shooter**",
            "🏢 Publisher": "Compania care a publicat jocul **Nintendo,Activision,Ubisoft**",
            "🌎 NA Sales": "Vânzări în America de Nord: **milioane de unități**.",
            "🇪🇺 EU Sales": "Vânzări în Europa: **milioane de unități**.",
            "🇯🇵 JP Sales": "Vânzări în Japonia: **milioane de unități**.",
            "🌍 Other Sales": "Vânzări în restul lumii: **milioane de unități**.",
            "📈 Global Sales": "Vânzări globale totale: **milioane de unități.** "
        }
        for nume, descriere in date_joc.items():
            analiza_joc(nume, descriere)
    if pagina_selectata == "Date lipsa":
        st.header("Date lipsa")
        coloana_selectata = st.selectbox("Alege o coloana", st.session_state['filtered_df'].columns)
        mesaj, numar_valori_lipsa = contoriazare_valori_lipsa(coloana_selectata)
        st.text(mesaj)

        if numar_valori_lipsa > 0 and pd.api.types.is_numeric_dtype(st.session_state['filtered_df'][coloana_selectata]):
            st.subheader("Alege o metoda de adaugare a valorilor lipsa")
            st.button("Metoda mediei", on_click=onClick, args=("mediei", coloana_selectata))
            st.button("Metoda modului", on_click=onClick, args=("modului", coloana_selectata))
            st.button("Metoda medianei", on_click=onClick, args=("medianei", coloana_selectata))

    if pagina_selectata == "Functii de grup":
        selected_column = st.selectbox("Coloana pentru grupare", df.columns)
        numerice = []
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                numerice += [column]
        selected_columns = st.multiselect("Coloana rezultate", numerice)
        operations = ['sum', 'mean', 'min', 'max', 'median', 'count', 'std', 'var']
        selected_operations = {}
        for col in selected_columns:
            selected_operations[col] = st.multiselect(f"Select operations for {col}", operations, key=col)
        agg_dict = {col: selected_operations[col] for col in selected_operations if selected_operations[col]}
        if agg_dict:
            df_grupat = df.groupby(selected_column).agg(agg_dict)
            filtru = st.text_input('Search by groupby column')
            df_filtrat = filtruNume(filtru, df_grupat)
            st.dataframe(df_filtrat)
        else:
            st.write("Please select at least one column and one operation.")
    if pagina_selectata == "Grafice":
        st.header("Selecteaza tipul de grafic dorit")
        grafice = ['Pie']
        grafic_selectat = st.selectbox("Grafice", grafice)
        if grafic_selectat == "Pie":
            pieChartUI()

def filtruNume(filtru, df):
    if filtru:  # Check if filter is not empty
        return df[df.index.astype(str).str.contains(filtru, case=False, na=False)]
    return df

def filtrare(column):
    filter_dict[column] = st.session_state.get(column, "")
    st.session_state['filtered_df'] = filter_dataframe()


def filter_dataframe():
    filtered_copy = df.copy()
    for column, value in filter_dict.items():
        if column in df.columns and value != "":
            if pd.api.types.is_numeric_dtype(filtered_copy[column]):
                condition = st.session_state.get(column + ' box', 'Equal')
                if condition == 'Bigger':
                    filtered_copy = filtered_copy[filtered_copy[column] > float(value)]
                elif condition == 'Smaller':
                    filtered_copy = filtered_copy[filtered_copy[column] < float(value)]
                else:
                    filtered_copy = filtered_copy[filtered_copy[column] == float(value)]
            elif pd.api.types.is_string_dtype(filtered_copy[column]):
                filtered_copy = filtered_copy[
                    filtered_copy[column].astype(str).str.contains(value, case=False, na=False)]

    return filtered_copy


def resetare():
    for column in st.session_state['filtered_df'].columns:
        st.session_state[column] = ""
    for (key, value) in filter_dict.items():
        filter_dict[key] = ""
    st.session_state['filtered_df'] = filter_dataframe()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.session_state[column + ' box'] = 'Equal'


def contoriazare_valori_lipsa(column_name):
    return f"Coloana  '{column_name}' contine {st.session_state['filtered_df'][column_name].isnull().sum()} valori lipsa", \
        st.session_state['filtered_df'][
            column_name].isnull().sum()


def onClick(nume, coloana_selectatat):
    val = 0
    if nume == "mediei":
        val = int(np.nanmean(st.session_state['filtered_df'][coloana_selectatat]))
    if nume == "modului":
        val = int(st.session_state['filtered_df'][coloana_selectatat].mode())
    if nume == "medianei":
        val = int(st.session_state['filtered_df'][coloana_selectatat].median())

    st.session_state['filtered_df'][coloana_selectatat] = st.session_state['filtered_df'][coloana_selectatat].fillna(
        val)
    if st.session_state['filtered_df'][coloana_selectatat].isnull().sum() > 0:
        st.markdown(f"<span style='color:red;'>Nu s-a actualizat cu succes!</span>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<span style='color:green;'>Am actualizat cu succes coloana {coloana_selectatat} folosind metoda {nume} -> {val}.</span>",
            unsafe_allow_html=True)

def pieChartUI():
    chart_columns = ['Name', 'Platform', 'Year', 'Genre', 'Publisher']
    result_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    column1 = st.selectbox('Coloana 1', chart_columns)
    column2 = st.selectbox('Coloana 2', result_columns)
    df_grupat = df.groupby(column1)[column2].sum()
    procente = (df_grupat / df_grupat.sum()) * 100
    fig, ax = plt.subplots()
    ax.pie(df_grupat, labels=df_grupat.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title(f'Distribuția {column2} per {column1}')

    st.pyplot(fig)