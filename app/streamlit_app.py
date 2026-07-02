from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_real_estate_with_clusters.csv"
STATS_PATH = ROOT / "data" / "processed" / "math_statistics.csv"
CLUSTER_PATH = ROOT / "data" / "processed" / "cluster_summary.csv"

st.set_page_config(page_title="Real Estate Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("Визуализация и анализ рынка недвижимости")
st.write("Проект по визуализации данных на основе Ames Housing test dataset. Так как в test.csv нет SalePrice, ML-задача реализована как кластеризация объектов недвижимости.")

st.sidebar.header("Фильтры")
neighborhoods = sorted(df["Neighborhood"].unique())
selected_neighborhoods = st.sidebar.multiselect("Район", neighborhoods, default=neighborhoods[:8])
year_min, year_max = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
selected_years = st.sidebar.slider("Год постройки", year_min, year_max, (year_min, year_max))
area_min, area_max = int(df["GrLivArea"].min()), int(df["GrLivArea"].max())
selected_area = st.sidebar.slider("Жилая площадь", area_min, area_max, (area_min, area_max))
clusters = sorted(df["Cluster"].unique())
selected_clusters = st.sidebar.multiselect("Кластер", clusters, default=clusters)
top_n = st.sidebar.slider("Top-N районов", 5, 20, 10)

filtered = df[
    (df["Neighborhood"].isin(selected_neighborhoods)) &
    (df["YearBuilt"].between(selected_years[0], selected_years[1])) &
    (df["GrLivArea"].between(selected_area[0], selected_area[1])) &
    (df["Cluster"].isin(selected_clusters))
]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Объектов", len(filtered))
k2.metric("Средняя жилая площадь", round(filtered["GrLivArea"].mean(), 1) if len(filtered) else 0)
k3.metric("Средний участок", round(filtered["LotArea"].mean(), 1) if len(filtered) else 0)
k4.metric("Среднее качество", round(filtered["OverallQual"].mean(), 2) if len(filtered) else 0)
k5.metric("Средний год", round(filtered["YearBuilt"].mean(), 0) if len(filtered) else 0)

st.subheader("Интерактивные визуализации")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.histogram(filtered, x="GrLivArea", nbins=30, title="Распределение жилой площади"), use_container_width=True)
with col2:
    top = filtered.groupby("Neighborhood", as_index=False)["GrLivArea"].mean().sort_values("GrLivArea", ascending=False).head(top_n)
    st.plotly_chart(px.bar(top, x="Neighborhood", y="GrLivArea", title="Средняя жилая площадь по районам"), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(px.box(filtered, x="OverallQual", y="GrLivArea", title="Площадь по уровню качества"), use_container_width=True)
with col4:
    st.plotly_chart(px.scatter(filtered, x="GarageArea", y="GrLivArea", color="Cluster", hover_data=["Neighborhood", "YearBuilt"], title="Связь гаража и жилой площади"), use_container_width=True)

st.subheader("Machine Learning Model and Mathematical Analysis")
st.write("ML-задача: KMeans-кластеризация объектов недвижимости по числовым характеристикам дома, участка, качества, года постройки и гаража.")
st.plotly_chart(px.scatter(filtered, x="PCA1", y="PCA2", color="Cluster", hover_data=["Neighborhood", "GrLivArea", "OverallQual"], title="Кластеры недвижимости в PCA 2D"), use_container_width=True)

st.write("Математико-статистический анализ")
st.dataframe(pd.read_csv(STATS_PATH), use_container_width=True)

st.write("Интерпретация кластеров")
st.dataframe(pd.read_csv(CLUSTER_PATH), use_container_width=True)

st.subheader("Краткие выводы")
st.markdown("""
- Большая жилая площадь чаще связана с более высоким уровнем качества объекта.
- Районы отличаются по средней площади и характеристикам недвижимости.
- Кластеризация разделяет дома на группы по размеру, качеству, году постройки и гаражу.
- В исходном test.csv нет целевой переменной SalePrice, поэтому прогноз цены невозможен без train.csv.
""")
