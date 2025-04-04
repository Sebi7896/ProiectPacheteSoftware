import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd

df = pd.read_csv('vgsales.csv', index_col=0)
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = df.copy(False)
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = pd.read_csv('GameReviews.csv', index_col=0)
filter_dict = {x: "" for x in df.columns}


def analiza_joc(nume, descriere):
    st.subheader(nume)
    st.markdown(descriere)


def script_sidebar(pagina_selectata):
    if pagina_selectata == "Acasa":
        st.header("🏠 Acasa")
        st.subheader("Dataframe-ul")
        col1, col2 = st.columns([1, 2])

        with col1:
            for column in st.session_state['filtered_df'].columns:
                if pd.api.types.is_numeric_dtype(st.session_state['filtered_df'][column].dtype):
                    st.text_input(label=column, max_chars=20, key=column, on_change=filtrare, args=[column])
                    st.selectbox(label=column, key=column + ' box', options=['Equal', 'Bigger', 'Smaller'],
                                 label_visibility='collapsed', on_change=filtrare(column))
                else:
                    st.text_input(label=column, max_chars=20, key=column, args=[column], on_change=filtrare)
            st.button(label="Resetare campuri", on_click=resetare)
        with col2:
            st.dataframe(st.session_state['filtered_df'], height=1400)

    if pagina_selectata == "Detalii Variabile":
        st.header("Detalii Variabile")
        st.dataframe(st.session_state['filtered_df'].describe())
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
        selected_column = st.selectbox("Coloana pentru grupare", st.session_state['filtered_df'].columns)
        numerice = []
        for column in st.session_state['filtered_df'].columns:
            if pd.api.types.is_numeric_dtype(st.session_state['filtered_df'][column]):
                numerice += [column]
        selected_columns = st.multiselect("Coloana rezultate", numerice)
        operations = ['sum', 'mean', 'min', 'max', 'median', 'count', 'std', 'var']
        selected_operations = {}
        for col in selected_columns:
            selected_operations[col] = st.multiselect(f"Select operations for {col}", operations, key=col)
        agg_dict = {col: selected_operations[col] for col in selected_operations if selected_operations[col]}
        if agg_dict:
            df_grupat = st.session_state['filtered_df'].groupby(selected_column).agg(agg_dict)
            filtru = st.text_input('Search by groupby column')
            df_filtrat = filtruNume(filtru, df_grupat)
            st.dataframe(df_filtrat)
        else:
            st.write("Please select at least one column and one operation.")
    if pagina_selectata == "Grafice":
        st.header("Selecteaza tipul de grafic dorit")
        grafice = ['Pie', 'Histogram']
        grafic_selectat = st.selectbox("Grafice", grafice)
        if grafic_selectat == "Pie":
            pieChartUI()
        elif grafic_selectat == "Histogram":
            histogram()
    if pagina_selectata == "Valori aberante":
        st.header("Valori aberante")
        numerice = [col for col in st.session_state['filtered_df'].columns
                    if pd.api.types.is_numeric_dtype(st.session_state['filtered_df'][col])]
        choosedColumn = st.selectbox(label="Selecteaza o coloana numerica", options=numerice, key=numerice)

        draw_boxplot(choosedColumn, st.session_state['filtered_df'])

    if pagina_selectata == "Scalarea datelor":
        st.header("Scalarea datelor")
        alegere = st.selectbox("Selecteaza metoda de scalare", ['Standardizare', 'Normalizare'])
        coloane_numerice = [col for col in st.session_state['filtered_df'].columns
                            if pd.api.types.is_numeric_dtype(st.session_state['filtered_df'][col]) and col != 'Year']

        if st.button("Aplica"):
            standardizare(coloane_numerice) if alegere == 'Standardizare' else normalizare(coloane_numerice)

        st.button("Resetare", on_click=resetare)
    if pagina_selectata == "GeoPandas":
        draw_map()
    if pagina_selectata == "Ratinguri":
        valori = ['Abysmalde', 'Terrible', 'Bad', 'Poor', 'Mediocre', 'Fair', 'Good', 'Great', 'Superb', 'Essential']
        st.dataframe(st.session_state['ratings'])
        if st.button("Codificare date"):
            st.session_state['ratings'] = codificare(valori, st.session_state['ratings'])
            st.dataframe(st.session_state['ratings'].describe())


def codificare(valori, df_ratinguri):
    df_ratinguri['Ratings'] = pd.Categorical(df_ratinguri['Review'], categories=valori, ordered=True).codes
    df_ratinguri['Ratings'] = np.interp(df_ratinguri['Ratings'], [0, len(valori) - 1], [0, 10])
    return df_ratinguri


def standardizare(coloane_numerice):
    scaler = StandardScaler()
    st.session_state['filtered_df'][coloane_numerice] = pd.DataFrame(
        scaler.fit_transform(st.session_state['filtered_df'][coloane_numerice]))
    st.text('Standardizare cu succes!')


def normalizare(coloane_numerice):
    min_max_scaler = MinMaxScaler()
    st.session_state['filtered_df'][coloane_numerice] = pd.DataFrame(
        min_max_scaler.fit_transform(st.session_state['filtered_df'][coloane_numerice]))
    st.text('Normalizare cu succes!')


def filtruNume(filtru, df):
    if filtru:  # Check if filter is not empty
        return df[df.index.astype(str).str.contains(filtru, case=False, na=False)]
    return df


def filtrare(column):
    filter_dict[column] = st.session_state.get(column, "")
    st.session_state['filtered_df'] = filter_dataframe()


def filter_dataframe():
    filtered_copy = st.session_state['filtered_df']
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
    st.session_state['filtered_df'] = df.copy()


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
    chart_columns = ['Platform', 'Year', 'Genre', 'Publisher']
    result_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    column1 = st.selectbox('Coloana 1', chart_columns)
    column2 = st.selectbox('Coloana 2', result_columns)

    df_grupat = df.groupby(column1)[column2].sum().sort_values(ascending=False)

    top_10 = df_grupat.head(10)
    restul = df_grupat.iloc[10:].sum()

    if restul > 0:
        top_10["Others"] = restul

    fig, ax = plt.subplots()
    ax.pie(top_10, labels=top_10.index, startangle=90, autopct='%1.1f%%')
    ax.set_title(f'Distribuția {column2} per {column1}')

    st.pyplot(fig)


def histogram():
    selected_column = st.selectbox("Alege o coloana dupa care sa ai distributia",
                                   ['Platform', 'Year', 'Genre', 'Publisher'])

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df[selected_column].dropna(), bins=20, color='skyblue', edgecolor='black')  # Drop NaNs
        ax.set_title(f'Distribuția {selected_column} pe frecvențe')
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frecvența")
        st.pyplot(fig)

    else:
        value_counts = df[selected_column].dropna().value_counts().sort_values(ascending=False)[:20]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(value_counts.index, value_counts.values, color='cornflowerblue')
        ax.set_title(f'Distribuția {selected_column} pe frecvențe')
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Număr de apariții")
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        st.pyplot(fig)


def draw_boxplot(column_name: str, df1: pd.DataFrame):
    min_val, max_val = int(df1[column_name].min()), int(df1[column_name].max())
    selected_range = st.slider("Select Sales Range", min_value=min_val, max_value=max_val, value=(min_val, max_val))
    filtered = df1[(df1[column_name] >= selected_range[0]) & (df1[column_name] <= selected_range[1] + 1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(filtered[column_name].dropna(), vert=True, patch_artist=True)
    ax.set_title(f'Boxplot of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Values')
    ax.grid(True)
    st.pyplot(fig)

    if st.button('Aplica schimbarile'):
        st.success(f"Filtrul a fost aplicat! {len(df)} rânduri selectate.")
        st.session_state['filtered_df'] = filtered


def draw_map():
    # Load the data for Steam users (Country, Value in millions)
    i_steam_df = pd.read_csv("./dateGeo.csv")

    # Load the Europe map (GeoJSON)
    europe = gpd.read_file("europe.json", encoding='latin-1')

    # Calculate area of each country in km² (if needed for visualization)
    europe["area_km2"] = europe.geometry.area / 1_000_000

    # Ensure that the coordinate reference system (CRS) is in EPSG 4326
    europe.to_crs(epsg=4326)
    europe = europe.merge(i_steam_df, left_on="name", right_on="Country", how="left")

    europe["Value"] = europe["Value"].fillna(0)
    # Generate the map with user data, using the `Value` column to color the countries
    europe = europe.rename(columns={"Value": "Values (millions)"})

    m = europe.explore(
        column="Values (millions)",
        legend=True,
        tooltip=["Values (millions)"],
        popup=["Values (millions)"],
        color_kwds={"cmap": "YlOrRd"}
    )

    # Save the map to an HTML file
    m.save("map.html")

    # Display the map in Streamlit
    st.header("Numar utilizatori Steam pe tara")
    st.components.v1.html(open("map.html", "r").read(), height=600, scrolling=True)
