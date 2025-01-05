import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load GeoJSON data
@st.cache_data
def load_geojson():
    geojson_path = 'ADMINISTRASIDESA_with_corrected_IDM.geojson'
    return gpd.read_file(geojson_path)

gdf = load_geojson()

# Load prediction data
@st.cache_data
def load_idm_data():
    idm_excel_path = 'IDM.xlsx'
    return pd.read_excel(idm_excel_path, sheet_name='Sheet2')

idm_data = load_idm_data()

# Prepare data for neural network
features = ['IKS 2021', 'IKS 2022', 'IKS 2023', 'IKE 2023.1', 'IKE 2022', 'IKE 2023', 'IKL 2023.1', 'IKL 2022', 'IKL 2023']
labels = ['NILAI IDM 2023']

X = idm_data[features].fillna(0).values
y = idm_data[labels].fillna(0).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predict IDM for 2045
future_years = np.array([[2045]] * len(idm_data))
X_future = scaler.transform(idm_data[features].fillna(0).values)
future_predictions = model.predict(X_future)

# Add predictions to the data frame
idm_data['PRED_IDM_2045'] = future_predictions

# Assign status based on predicted IDM
def assign_status(idm):
    if idm >= 0.81:
        return "MANDIRI"
    elif idm >= 0.71:
        return "MAJU"
    elif idm >= 0.61:
        return "BERKEMBANG"
    else:
        return "TERTINGGAL"

idm_data['PRED_STATUS_2045'] = idm_data['PRED_IDM_2045'].apply(assign_status)

# Merge predictions with GeoJSON
gdf = gdf.merge(idm_data[['DESA', 'PRED_IDM_2045', 'PRED_STATUS_2045']], left_on='NAMOBJ', right_on='DESA', how='left')

gdf['latitude'] = gdf.geometry.centroid.y
gdf['longitude'] = gdf.geometry.centroid.x
gdf['color'] = gdf['PRED_IDM_2045'].apply(lambda x: [255 * (1 - x), 255 * x, 0, 160])

gdf['elevation'] = gdf['PRED_IDM_2045'] * 1000

# Streamlit app title
st.title("Prediksi IDM Desa di Kabupaten Malang Tahun 2045 dengan Neural Network")
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
st.markdown("### Legenda IDM Score Tahun 2045")
st.markdown("<span style='color: rgb(0, 255, 0);'>■</span> Nilai Tinggi", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(255, 0, 0);'>■</span> Nilai Rendah", unsafe_allow_html=True)
st.markdown("<span style='color: rgb(200, 200, 200);'>■</span> Tidak Tersedia", unsafe_allow_html=True)

st.markdown("Peta ini menggambarkan prediksi nilai IDM dan status desa di Kabupaten Malang untuk tahun 2045 berdasarkan neural network.")
