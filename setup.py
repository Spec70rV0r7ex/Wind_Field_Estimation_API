from setuptools import setup, find_packages

setup(
    name="sar_wind_retrieval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "earthengine-api",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn"
    ],
    author="Expert Geospatial Engineering Team",
    description="High-resolution ocean surface wind field estimation using SAR and CMOD5.N"
)