PLANT DISEASE DETECTION SYSTEM

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

3. Create and Activate Virtual Environment
   
python -m venv venv

venv\Scripts\activate

5. Install Dependencies
   
pip install -r requirements.txt

7. Run the Application
   
python app.py


The app will open in the browser

Upload a plant leaf image → get predicted disease & confidence score.


Output Screens

Login Page:

<img width="1491" height="844" alt="3ff541e874b42c870df573695192d412_login_page" src="https://github.com/user-attachments/assets/dd2e7380-5f45-4ca8-b0b9-dd6d7f91a10e" />

Home Page:

<img width="1298" height="618" alt="1cb7f3737c641cfac827d2def1e67d5e_home_page" src="https://github.com/user-attachments/assets/4e725f28-3323-4f2a-8be3-1be5c77e51f1" />

Prediction Screen:

<img width="1073" height="723" alt="image" src="https://github.com/user-attachments/assets/a4962fa7-9cb4-4202-8532-811ef270e4eb" />

Accuracy Graph

<img width="718" height="564" alt="c16d325d0d3cda8aea0c5e09a605b25b_accuracy_graph" src="https://github.com/user-attachments/assets/736bbeb8-2279-42f0-ba17-a69454fa51fd" />

Confusion Matrix

<img width="755" height="834" alt="3eec4f46f6ab85fd92cab196dfeaadac_confusion_matrix" src="https://github.com/user-attachments/assets/2d172007-cf30-4217-817c-edea5c1d2c0e" />


System Architecture :

User uploads leaf image

Preprocessing module prepares image

CNN model extracts features

Classifier predicts disease

UI displays output


Conclusion :

The Plant Disease Detection System successfully identifies plant diseases with high accuracy using deep learning. It reduces the dependency on manual inspection and supports early detection, better treatment, and increased crop yield.


Future Enhancements :

Mobile application deployment

Real-time camera-based detection

Enlarged dataset for higher accuracy

Support for more plant species

Bounding box detection for infection regions


References :

TensorFlow Documentation

Keras API

PlantVillage Dataset

Google Colab

Deep Learning Research Papers
