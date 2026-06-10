import streamlit as st
import requests
import os
import datetime
from PIL import Image

st.set_page_config(
    page_title = "SAR Wind Dashboard", 
    page_icon = "🌪️", 
    layout = "wide"
)

st.sidebar.title("⚙️ Pipeline Parameters")
st.sidebar.markdown("Configure the spatial and temporal parameters for the SAR wind retrieval.")

target_date = st.sidebar.date_input("Select Target Date", value = datetime.date(2024, 2, 9))

st.sidebar.subheader("Bounding Box (Decimal Degrees)")
lon_min = st.sidebar.number_input("Min Longitude", value = 67.0, step = 0.1)
lon_max = st.sidebar.number_input("Max Longitude", value = 72.5, step = 0.1)
lat_min = st.sidebar.number_input("Min Latitude", value = 20.0, step = 0.1)
lat_max = st.sidebar.number_input("Max Latitude", value = 24.0, step = 0.1)

polygon = [
    [lon_min, lat_min],
    [lon_max, lat_min],
    [lon_max, lat_max],
    [lon_min, lat_max],
    [lon_min, lat_min]
]

st.title("SAR Coastal Wind Retrieval Dashboard")
st.markdown("""
This application interfaces with the pipeline's FastAPI backend to retrieve Sentinel-1 SAR imagery from Google Earth Engine, 
process it through the M64RN4 Deep Residual Network, and output a high-resolution, operationally interpolated wind field using CMOD5 inversion.
""")

st.divider()

if st.button("Execute SAR Pipeline", type = "primary"):
    with st.status("Running Wind Retrieval Pipeline...", expanded = True) as status:
        st.write("Pinging FastAPI backend (http://127.0.0.1:8000)...")
        url = "http://127.0.0.1:8000/api/v1/wind-field"

        payload = {
            "date": target_date.strftime("%Y-%m-%d"),
            "polygon": polygon
        }

        try:
            response = requests.post(url, json = payload)            
            if response.status_code == 200:
                st.write("✅ SAR extraction and neural network processing complete!")
                st.write("🗺️ Interpolating grid and generating Cartopy visualization...")
                json_filename = f"data/gujarat_wind_{target_date.strftime('%Y-%m-%d')}.json"
                output_image = "output_map.png"
                os.system(f"python run_plot.py {json_filename} {output_image}")
                status.update(label = "Pipeline Complete!", state = "complete", expanded = False)
                
                if os.path.exists(output_image):
                    st.success(f"Successfully generated operational map for {target_date.strftime('%Y-%m-%d')}.")
                    image = Image.open(output_image)
                    st.image(image, use_container_width = True)
                else:
                    st.error("Map generation failed. Please check the terminal for `run_plot.py` errors.")
            else:
                status.update(label = "Pipeline Error", state = "error", expanded = True)
                st.error(f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            status.update(label = "Connection Failed", state = "error", expanded = True)
            st.error("Failed to connect to the backend. Please ensure your FastAPI server is running in a separate terminal: `uvicorn api.main:app --reload`")