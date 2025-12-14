from setuptools import setup, find_packages

setup(
    name="nba_score_winpred",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "requests",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "pydantic",
        "nba_api",
    ],
)
