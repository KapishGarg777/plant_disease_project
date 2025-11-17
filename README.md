Plant Disease Detection System
Deep Learning–based Image Classification

This project uses a Convolutional Neural Network (CNN) to automatically detect and classify diseases in plant leaves. It helps farmers, researchers, and agricultural experts diagnose plant diseases early using image-based prediction.

Features :
 Automatic detection of plant leaf diseases
 CNN-based highly accurate deep learning model
 Upload image → get instant prediction
 98.10% training accuracy & 95.07% validation accuracy
 Automatic image preprocessing
 Built and trained using Google Colab
 Simple user interface using Flask/Streamlit
 Supports multiple plant disease classes

plant_disease_project/
│
├── app.py                      # Main application file
├── trained_model.h5            # Trained CNN model (~90MB)
├── home_page.jpeg              # UI image
├── requirements.txt            # Python dependencies
├── train_plant_disease.ipynb   # Training notebook (uploaded manually)
├── test_plant_disease.ipynb    # Testing notebook (uploaded manually)
└── README.md                   # Project documentation

How to Run the Project :
1. Clone the Repository
git clone https://github.com/KapishGarg777/plant_disease_project.git
cd plant_disease_project

2. Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python app.py

The app will open in the browser

Upload a plant leaf image → get predicted disease & confidence score.

