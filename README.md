# High-Resolution SAR Coastal Wind Retrieval Pipeline

## 🎯 Project Aim
This pipeline automates the extraction and processing of high-resolution ocean surface wind fields from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. Designed to process data over complex coastal regions (like the Gujarat coast), the system leverages a ResNet neural network and the CMOD5 Geophysical Model Function (GMF) to invert raw radar backscatter into accurate meteorological wind vectors.

### Key Scientific Features:
- **Dynamic Earth Engine Mosaicking:** Automatically bypasses GEE's 10MB payload limits by dynamically shrinking bounding boxes, matching orbit passes, and extracting data via 2D flat rasters (`sampleRectangle`) rather than massive feature collections.
- **Morphological Coastal Buffering:** Utilizes `scipy.ndimage.binary_dilation` to apply a 1-pixel safety buffer along shorelines, eliminating the artificial hurricane-force wind artifacts typically caused by surf-zone radar brightness.
- **Empirical dB Calibration:** Applies a custom -5.0 dB calibration offset to raw VV backscatter to correct for regional Arabian Sea brightness anomalies prior to CMOD5 inversion.
- **Automated Data Persistence:** Intercepts calculated wind fields and silently auto-saves them as formatted `.json` files for immediate use in Jupyter Notebooks and `cartopy` visualizations.

---

## 📂 Project Structure
```text
├── api/
│   └── main.py                 # FastAPI server and CMOD5 inversion logic
├── gee/
│   └── fetch_sentinel1.py      # Earth Engine data extraction and local patch slicing
├── model/
│   └── postprocess.py          # Morphological land masking and ResNet logic
├── data/                       # Auto-generated JSON outputs drop here
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
## 🚀 Installation & Setup
1. Clone the repository and navigate to the directory:
git clone <your-repo-link>
cd <your-repo-name>
2. Create a virtual environment (Recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install the required dependencies:
pip install -r requirements.txt
4. Authenticate Google Earth Engine:
You must authenticate your local machine to pull satellite imagery from Google's servers. Run this command and follow the browser prompts:
earthengine authenticate

## ⚙️ Running the Pipeline
**Step 1: Start the Backend Server**

Initialize the FastAPI server. The --reload flag ensures the server automatically updates if you change the code.
uvicorn api.main:app --reload
The server will start at http://127.0.0.1:8000

**Step 2: Execute a Wind Field Retrieval**

You can trigger the pipeline in two ways:

**Option A: Using the Browser (Swagger UI)**

1. Open http://127.0.0.1:8000/docs in your web browser.
2. Navigate to the POST /get_wind_field endpoint.
3. Click "Try it out", enter your target date (e.g., "2024-02-09") and your bounding box polygon.
4. Click Execute.

**Option B: Using Python (Automated Bulk Processing)**

Run this script in a Jupyter Notebook to ping the API. The API will silently crunch the SAR data and drop the formatted .json file straight into your data/ folder.

import requests

url = "[http://127.0.0.1:8000/get_wind_field](http://127.0.0.1:8000/get_wind_field)"

payload = {
  "date": "2024-02-09",
  "polygon": [[67.0, 20.0],[72.5, 20.0],[72.5, 24.0],[67.0, 24.0],[67.0, 20.0]]
}

response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")

**Step 3: Visualize the Results**

Once the JSON is saved in your data/ directory, run your cartopy plotting script to generate the final, publication-ready quiver plot of the wind field.

## ⚠️ Known Limitations & Future Work
**Spatial Limits:**

Due to synchronous GEE payload caps, attempting to process an area larger than ~1.5° x 1.5° at 100m resolution in a single API call will result in a Computed value is too large error.

**Future Scaling:**

To map massive coastlines simultaneously, implement a dynamic spatial-tiling loop in your Jupyter Notebook to break the macro-bounding box into discrete 1° x 1° tiles before pinging the API.


***

**Final Tip:** Make sure to uncomment either `torch` or `tensorflow` in the `requirements.txt` file depending on which library you actually used to load and run your ResNet!
