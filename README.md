# High Resolution SAR Coastal Wind Retrieval Pipeline

## Project Aim

This pipeline automates the extraction and processing of high-resolution ocean surface wind fields from Sentinel-1 Synthetic Aperture Radar imagery. Designed specifically for complex coastal zones, the system leverages a PyTorch-based Deep Residual Network and a modified CMOD5 Geophysical Model Function to invert raw radar backscatter into accurate meteorological wind vector fields.

### Key Scientific & Engineering Features

* **Dynamic Earth Engine Mosaicking:** Programmatically bypasses Google Earth Engine's 10 MB payload restrictions. The pipeline uses flat 2D raster extractions via `sampleRectangle` with dynamic spatial tiling and strict temporal matching to prevent memory allocation failures over long swaths.
* **Morphological Coastal Buffering:** Employs `scipy.ndimage.binary_dilation` to apply a dynamic 1 pixel safety buffer along shorelines. This physically isolates and removes wave-breaking surf zones and bright intertidal mudflats, eliminating false hurricane force wind artifacts.
* **Empirical dB Space Calibration:** Applies a custom -4.0 dB calibration offset to the raw VV backscatter to correct for persistent baseline brightness over the Arabian Sea. The inversion engine operates natively in logarithmic space to suppress exponential speed overestimations near the swath edges.
* **Automated Data Persistence:** Intercepts computed wind fields and automatically serializes them into structured `.json` files for seamless downstream integration with Jupyter Notebooks, `cartopy` plotting scripts, or web dashboards.

---

## 📂 Project Structure

```text
├── api/
│   ├── main.py                 # FastAPI server and CMOD5 inversion logic
│   └── schemas.py              # Data validation and Pydantic API schemas
├── config/
│   └── settings.py             # Global configurations and GEE environment tokens
├── data/                       # Auto-generated JSON pipeline outputs
│   └── gujarat_wind_...json    # Example serialized wind vector data
├── gee/
│   └── fetch_sentinel1.py      # Earth Engine data extraction and patch slicing
├── model/
│   ├── weights/                # Saved PyTorch model weights (.pth)
│   ├── postprocess.py          # Morphological land masking and de-aliasing logic
│   ├── predict.py              # ResNet inference routines
│   ├── resnet.py               # PyTorch M64RN4 architecture definition
│   └── train.py                # Model training loop and dataset wrappers
├── notebooks/
│   └── demo_gujarat.py         # Script to plot raw SAR-derived wind vector fields
├── utils/
│   ├── preprocessing.py        # Normalization and quality control utilities
│   ├── validation_math.py      # Validation metrics
│   ├── visualization.py        # Cartopy and Matplotlib mapping wrappers
│   └── windspeed.py            # Core wind vector and GMF mathematical equations
├── app.py                      # Interactive Streamlit frontend dashboard
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── run_plot.py                 # Grid interpolation and continuous surface mapping script

```

---

## Installation & Setup

**1. Clone the Repository**

```bash
git clone <your-repo-link>
cd <your-repo-name>

```

**2. Configure the Python Environment**
Ensure you are using Python 3.11 within a Conda virtual environment for geospatial dependency stability:

```bash
conda activate venv
pip install -r requirements.txt

```

**3. Authenticate Google Earth Engine**
You must authorize your machine to access Google's satellite image catalogs. Run the following command and complete the authentication handshake via your web browser:

```bash
earthengine authenticate

```

---

## Running the Pipeline

To run the full end-to-end interactive stack, you will need to open two separate terminal sessions.

### Step 1: Initialize the FastAPI Backend Server

In your first terminal tab, start the processing engine using Uvicorn. The `--reload` flag ensures code adjustments are synced in real-time.

```bash
uvicorn api.main:app --reload

```

*The backend server will instantiate locally at `http://127.0.0.1:8000`.*

### Step 2: Launch the Interactive Streamlit Frontend Dashboard

In a second terminal tab, launch the web interface to easily run queries and visualize predictions:

```bash
streamlit run app.py

```

*Your browser will automatically open `http://localhost:8501` to display the interactive UI.*

### Alternative: Programmatic API Execution (Python)

If you prefer to bypass the user interfaces entirely, you can dispatch queries to the pipeline directly via a Python script:

```python
import requests

url = "http://127.0.0.1:8000/api/v1/wind-field"

payload = {
    "date": "2024-02-09",
    "polygon": [[67.0, 20.0], [72.5, 20.0], [72.5, 24.0], [67.0, 24.0], [67.0, 20.0]]
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")

```

---

## Data Visualization & Output Formats

Once the pipeline outputs a successful execution payload, data is written directly to the `data/` directory. You can visualize the results using two distinct operational mapping frameworks:

* **Micro-Scale Vector Mapping:** Run `notebooks/demo_gujarat.py` to generate an unsmoothed, high resolution vector plot. This mapping technique isolates localized sub-mesoscale atmospheric variations, coastal wind shadowing, and raw convective wind streaks.
* **Continuous Surface Mapping:** Run the `run_plot.py` script to apply cubic grid interpolation across the sparse data points. This creates an operational meteorological field chart complete with continuous color gradients:
```bash
python run_plot.py data/gujarat_wind_2024-02-09.json output_map.png

```



---

## Known Constraints & Operational Limits

* **GEE Payload Constraints:** Attempting to ingest bounding boxes broader than 1.5° × 1.5° at a strict 100 m pixel resolution within a single request will trigger an Earth Engine memory limit error (`Computed value is too large`).
* **Macro-Scale Processing Workaround:** To map the entirety of the 600 km Gujarat coastline simultaneously without exhausting server memory allocation, partition large regions into discrete 1° × 1° spatial tiles, process each step sequentially through the API, and mosaic the resulting output arrays.
