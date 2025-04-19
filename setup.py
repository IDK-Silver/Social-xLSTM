# -*- coding: utf-8 -*-
# File location: setup.py (in your project root)

import setuptools
import os

# Function to read the README file for the long description
def read_readme(fname):
    """Safely read the README file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {fname} not found.")
        return "" # Return empty string if README is not found

# --- Configuration ---
# TODO: Review and update remaining placeholders (VERSION, DESCRIPTION, LICENSE_TYPE, etc.)

# Distribution Name - How it will be known on PyPI or when installed (hyphens preferred)
PACKAGE_NAME = "social-xlstm"
# The actual Python package directory name under 'src/' (currently 'dataset')
# NOTE: This differs from PACKAGE_NAME above. Consider renaming 'src/dataset' to 'src/social_xlstm'
# for consistency. If you rename the directory, update your import statements accordingly.
PYTHON_PACKAGE_DIR = "dataset"
VERSION = "0.1.0" # TODO: Update version as needed
AUTHOR = "Yu-Feng Huang" # Updated
AUTHOR_EMAIL = "a288235403@gmail.com" # Updated
DESCRIPTION = "A package to process datasets from zipped XML files for PyTorch (Social-xLSTM related)." # TODO: Refine description
URL = "https://github.com/IDK-Silver/Social-xLSTM" # Updated
# License type - Common examples: 'MIT', 'Apache-2.0', 'GPL-3.0-only'
# Ensure this matches your LICENSE file content.
LICENSE_TYPE = "MIT" # TODO: Change to your actual license type if different

# Define runtime dependencies required by your package
# TODO: Sync this list with your environment.yaml (core runtime dependencies)
# Exclude development/testing tools like pytest here.
INSTALL_REQUIRES = [
    'torch',
    # Add other libraries needed to RUN your code, e.g.,
    # 'lxml',
    # 'numpy',
]

# --- Setup Call ---

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_readme("README.md"), # Read README for long description
    long_description_content_type="text/markdown", # Specify README format
    url=URL, # Project URL
    license=LICENSE_TYPE, # Specify the license type string

    # === Crucial for src/ layout ===
    package_dir={'': 'src'}, # Tell setuptools that packages are under the 'src' directory
    # Automatically finds packages in 'src'. Currently finds 'dataset'.
    # If you rename src/dataset to src/social_xlstm, it will find 'social_xlstm'.
    packages=setuptools.find_packages(where='src'),
    # ================================

    # Specify Python versions your code supports.
    python_requires='>=3.8', # TODO: Adjust based on your minimum supported Python version

    # List of dependencies required to run the package
    install_requires=INSTALL_REQUIRES,

    # Include package data (non-code files within your package) if any
    # include_package_data=True,
    # package_data={
    #     # Example: 'your_package_name': ['data/*.csv'],
    # },

    # Add classifiers to help users find your project; see https://pypi.org/classifiers/
    classifiers=[
        # TODO: Review and adjust classifiers
        "Development Status :: 3 - Alpha", # Initial development stage
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        f"License :: OSI Approved :: {LICENSE_TYPE} License", # Reflects the LICENSE_TYPE above
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", # Add versions you support/test
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Optional: Entry points for command-line scripts
    # entry_points={
    #     'console_scripts': [
    #         'your_script_name=your_package_name.module:main_function',
    #     ],
    # },
)

print("-" * 60)
print(f"Setup configuration for {PACKAGE_NAME} version {VERSION}")
# This will print ['dataset'] if your directory is src/dataset
print("Packages found:", setuptools.find_packages(where='src'))
print("-" * 60)