from setuptools import setup, find_packages

setup(
    name="earnings_transcript_predictor",
    version="0.1.0",
    author="Shannon Maccallum",
    description="Predicting stock returns from earnings call transcripts using deep learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "yfinance>=0.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
        "python-dotenv>=1.0.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.6.0",
    ],
)
