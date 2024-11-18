# FixU Project

## Overview
The FixU project aims to develop a comprehensive solution for analyzing and predicting mental health conditions such as depression among different user groups, including students and working professionals. The project leverages machine learning models to provide insights and predictions based on user data.

## Repository Structure
The repository is organized into the following main directories:

- **fixu_professional**: Contains notebooks and scripts related to the analysis and modeling for working professionals.
- **fixu_student**: Contains notebooks and scripts related to the analysis and modeling for students.
- **web app**: Contains the web application code, including HTML, CSS, and JavaScript files.

## Notebooks
### fixu_professional/notebooks/fixu_main.ipynb
This notebook includes:
- Data gathering and preprocessing
- Data mapping and transformation
- Model training and evaluation for working professionals

### fixu_student/notebooks/fixu_main.ipynb
This notebook includes:
- Data gathering and preprocessing
- Data mapping and transformation
- Model training and evaluation for students

## Requirements
The project requires several Python packages. The dependencies are listed in the `requirements.txt` files located in the `fixu_professional` and `fixu_student` directories.

### Installation
To install the required packages, run:
```bash
pip install -r fixu_professional/requirements.txt
pip install -r fixu_student/requirements.txt
```

## Web Application
The web application provides an interface for users to input their data and receive predictions. The main files include:
- `index.html`: The main page where users can choose their user type (student or professional).
- `styles.css`: The stylesheet for the web application.

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fixu.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fixu
    ```
3. Install the required packages:
    ```bash
    pip install -r fixu_professional/requirements.txt
    pip install -r fixu_student/requirements.txt
    ```
4. Run the notebooks to preprocess data and train models.
5. Launch the web application to interact with the models.


