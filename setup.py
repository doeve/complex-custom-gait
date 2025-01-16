# File: setup.py

from setuptools import setup, find_packages

setup(
    name="gait_recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.16.1",
        "numpy>=1.23.5",
        "opencv-python>=4.6.0",
        "Pillow>=9.3.0",
        "pyyaml>=6.0",
        "pandas>=1.5.2",
        "h5py>=3.7.0",
        "typing-extensions>=4.4.0",
        "dataclasses>=0.6",
        "tqdm>=4.64.1",
        "loguru>=0.6.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.2.0',
            'black>=22.3.0',
            'isort>=5.10.1',
            'flake8>=5.0.4',
            'matplotlib>=3.6.2',
            'seaborn>=0.12.1',
            'jupyter>=1.0.0',
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Gait recognition system using deep learning",
    keywords="gait recognition, deep learning, computer vision",
    python_requires=">=3.10",
)