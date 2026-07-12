import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Données d'exemple
data = pd.DataFrame({
    "Heure": [6, 7, 8, 9, 10, 16, 17, 18, 19],
    "Jour": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Voyageurs": [120, 250, 450, 300, 180, 350, 520, 480, 260]
})

X = data[["Heure", "Jour"]]
y = data["Voyageurs"]

modele = LinearRegression()
modele.fit(X, y)

st.title("Prévision des voyageurs - ETO Oran")

heure = st.slider("Heure", 0, 23, 8)
jour = st.selectbox(
    "Jour de la semaine",
    [1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: [
        "Lundi", "Mardi", "Mercredi",
        "Jeudi", "Vendredi", "Samedi", "Dimanche"
    ][x - 1]
)

prediction = modele.predict([[heure, jour]])[0]

st.metric("Nombre prévu de voyageurs", f"{int(prediction)}")

capacite_bus = 80
nb_bus = max(1, int(prediction / capacite_bus) + 1)

st.success(f"Nombre de bus recommandé : {nb_bus}")

st.subheader("Données utilisées")
st.dataframe(data)
