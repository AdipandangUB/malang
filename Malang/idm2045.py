import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import numpy as np

# Load GeoJSON data
@st.cache_data
def load_geojson():
    geojson_path = 'ADMINISTRASIDESA_with_corrected_IDM.geojson'
    return gpd.read_file(geojson_path)

gdf = load_geojson()

# Load prediction data
@st.cache_data
def load_predictions():
    predictions_path = 'idm2045.csv'  # Placeholder for uploaded prediction file
    return pd.read_csv(predictions_path)

predictions_df = load_predictions()

# Merge predictions with GeoJSON
gdf = gdf.merge(predictions_df, left_on='NAMOBJ', right_on='DESA', how='left')

# Define color mapping for IDM status
status_colors = {
    "MANDIRI": [0, 128, 0, 160],  # Dark Green
    "MAJU": [0, 255, 0, 160],  # Bright Green
    "BERKEMBANG": [255, 165, 0, 160],  # Orange
    "TERTINGGAL": [255, 0, 0, 160],  # Red
    "Tidak Tersedia": [200, 200, 200, 160],  # Gray
}

gdf['latitude'] = gdf.geometry.centroid.y
gdf['longitude'] = gdf.geometry.centroid.x
gdf['color'] = gdf['PRED_STATUS_2045'].map(status_colors)

gdf['elevation'] = gdf['PRED_IDM_2045'] * 1000

# Streamlit app title
st.title("Prediksi IDM Desa di Kabupaten Malang Tahun 2045")
st.header("Riset: Analisis Spasial Indeks Pembangunan Desa: Hubungan Antara Karakteristik Fisik Geografis dan Capaian Indeks Pembangunan Desa")
st.subheader("Oleh: Dr.GES.AR. Rohman Taufiq Hidayat,ST., M.AgrSc.; Adipandang Yudono, S.Si., MURP., Ph.D.; Wawargita Permata Wijayanti, ST., MT.")


# Display raw data
if st.checkbox("Tampilkan data mentah"):
    st.write(gdf[['NAMOBJ', 'PRED_STATUS_2045', 'PRED_IDM_2045', 'latitude', 'longitude']])

# Define the layer for Pydeck visualization
layer = pdk.Layer(
    "ColumnLayer",
    data=gdf,
    get_position=["longitude", "latitude"],
    get_elevation="elevation",
    elevation_scale=1,
    radius=500,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

# Define the initial view state for the map
view_state = pdk.ViewState(
    latitude=gdf['latitude'].mean(),
    longitude=gdf['longitude'].mean(),
    zoom=10,
    pitch=40,
)

# Create Pydeck chart
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Desa: {NAMOBJ}\nStatus: {PRED_STATUS_2045}\nNilai IDM 2045: {PRED_IDM_2045}"},
))

# Add a legend
st.markdown("### Legenda IDM Status Tahun 2045")
st.markdown("<span style='color: rgb(0, 128, 0);'>■</span> MANDIRI", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(0, 255, 0);'>■</span> MAJU", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(255, 165, 0);'>■</span> BERKEMBANG", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(255, 0, 0);'>■</span> TERTINGGAL", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(200, 200, 200);'>■</span> Tidak Tersedia", unsafe_allow_html=True)

st.markdown("Peta ini menggambarkan prediksi dengan Regresi nilai IDM dan status desa di Kabupaten Malang untuk tahun 2045 berdasarkan tren data historis.")
