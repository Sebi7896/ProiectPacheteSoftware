import streamlit as st


def analiza_joc(nume, descriere):
    st.subheader(nume)
    st.markdown(descriere)


def script_sidebar(pagina_selectata, df):
    if pagina_selectata == "Acasa":
        st.header("🏠 Acasă")
        st.subheader("Dataframe-ul")
        st.dataframe(df, height=500)
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
