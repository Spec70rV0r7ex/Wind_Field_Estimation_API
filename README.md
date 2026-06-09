# High Resolution SAR Coastal Wind Retrieval Pipeline

## Project Aim
This pipeline automates the extraction and processing of high-resolution ocean surface wind fields from Sentinel-1 Synthetic Aperture Radar imagery. Designed to process data over complex coastal regions, the system leverages a ResNet neural network and the CMOD5 Geophysical Model Function to invert raw radar backscatter into accurate meteorological wind vectors.

### Key Scientific Features:
- **Dynamic Earth Engine Mosaicking:** Automatically bypasses GEE's 10MB payload limits by dynamically shrinking bounding boxes, matching orbit passes, and extracting data via 2D flat rasters (`sampleRectangle`) rather than massive feature collections.
- **Morphological Coastal Buffering:** Utilizes `scipy.ndimage.binary_dilation` to apply a 1-pixel safety buffer along shorelines, eliminating the artificial hurricane-force wind artifacts typically caused by surf-zone radar brightness.
- **Empirical dB Calibration:** Applies a custom -4.0 dB calibration offset to raw VV backscatter to correct for regional Arabian Sea brightness anomalies prior to CMOD5 inversion.
- **Automated Data Persistence:** Intercepts calculated wind fields and silently auto-saves them as formatted `.json` files for immediate use in Jupyter Notebooks and `cartopy` visualizations.

---

## 📂 Project Structure
```text
├── api/
│   ├── main.py                 # FastAPI server and CMOD5 inversion logic
│   └── schemas.py              # Data validation and API schemas
├── config/
│   └── settings.py             # Global configurations and environment variables
├── data/                       # Auto-generated JSON outputs drop here
│   └── gujarat_wind_...json    # Example model output
├── gee/
│   └── fetch_sentinel1.py      # Earth Engine data extraction and patch slicing
├── model/
│   ├── weights/                # Stored PyTorch model weights
│   ├── postprocess.py          # Morphological land masking logic
│   ├── predict.py              # ResNet inference and prediction script
│   ├── resnet.py               # M64RN4 architecture definition
│   └── train.py                # Model training loop and dataset loading
├── notebooks/
│   └── demo_gujarat.py         # Script to plot the raw SAR-derived wind vector field
├── utils/
│   ├── preprocessing.py        # Data cleaning and normalization utilities
│   ├── validation_math.py      # MAE, RMSE, and Bias calculation logic
│   ├── visualization.py        # Helper functions for plotting
│   └── windspeed.py            # Core wind vector and speed math
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── run_plot.py                 # Script to plot the interpolated wind field surface map

```

---

## Installation & Setup

**1. Clone the repository and navigate to the directory:**

```bash
git clone <your-repo-link>
cd <your-repo-name>

```

**2. Create a virtual environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt

```

**4. Authenticate Google Earth Engine:**
You must authenticate your local machine to pull satellite imagery from Google's servers. Run this command and follow the browser prompts:

```bash
earthengine authenticate

```

---

## Running the Pipeline

### Step 1: Start the Backend Server

Initialize the FastAPI server. The `--reload` flag ensures the server automatically updates if you change the code.

```bash
uvicorn api.main:app --reload

```

*The server will start at `http://127.0.0.1:8000*`

### Step 2: Execute a Wind Field Retrieval

You can trigger the pipeline in two ways:

**Option A: Using the Browser (Swagger UI)**

1. Open `http://127.0.0.1:8000/docs` in your web browser.
2. Navigate to the `POST /get_wind_field` endpoint.
3. Click **"Try it out"**, enter your target date (e.g., `"2024-02-09"`) and your bounding box polygon.
4. Click **Execute**.

**Option B: Using Python**
Run this script to ping the API. The API will silently crunch the SAR data and drop the formatted `.json` file straight into your `data/` folder.

```python
import requests

url = "[http://127.0.0.1:8000/get_wind_field](http://127.0.0.1:8000/get_wind_field)"

payload = {
  "date": "2024-02-09",
  "polygon": [[67.0, 20.0],[72.5, 20.0],[72.5, 24.0],[67.0, 24.0],[67.0, 20.0]]
}

response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")

```

### Step 3: Visualize the Results

Once the JSON is saved in your `data/` directory, you can generate two distinct types of publication-ready maps depending on your analytical needs:

* **Raw Data Mapping:** Run `notebooks/demo_gujarat.py` to generate the **Raw SAR-derived wind vector field over Gujarat**. This plot accurately displays the un-smoothed, micro-scale wind vectors exactly as captured and predicted by the ResNet model.
* **Operational Mapping:** Run the `run_plot.py` script via your terminal to generate an **Interpolated wind field surface map of the Gulf of Kutch**. This applies a uniform interpolation grid to create a smooth, continuous background color gradient with perfectly spaced arrows.
```bash
python run_plot.py data/gujarat_wind_20240209.json output_map.png

```



---

## Known Limitations & Future Work

* **Spatial Limits:** Due to synchronous GEE payload caps, attempting to process an area larger than ~1.5° x 1.5° at 100m resolution in a single API call will result in a `Computed value is too large` error.
* **Future Scaling:** To map massive coastlines simultaneously, implement a dynamic spatial-tiling loop in your Jupyter Notebook to break the macro-bounding box into discrete 1° x 1° tiles before pinging the API.

---

**Final Tip:** Make sure to uncomment either `torch` or `tensorflow` in the `requirements.txt` file depending on which library you actually used to load and run your ResNet!
