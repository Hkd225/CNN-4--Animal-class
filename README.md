# Animals-10 Image Classification (Top 4 Classes)
## Transfer Learning Pipeline & Automated Submission Generator
### By Muhammad Auffa Hakim Aditya

This project presents a highly automated Deep Learning pipeline that filters specific animal classes from the Animals-10 dataset, trains a hybrid Custom CNN + MobileNetV2 classifier, and automatically packages all deployment artifacts (TFLite, TF.js, SavedModel) into a structured, submission-ready ZIP file.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate advanced Machine Learning Engineering workflows, focusing on automated data wrangling, two-stage transfer learning, and seamless preparation for cross-platform production environments.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automated Data Wrangling: Download the raw Animals-10 dataset and selectively extract only the 4 target classes to a clean directory.
2. Robust Data Splitting: Automatically divide the filtered data into 80% Training, 10% Validation, and 10% Testing sets using `split-folders`.
3. Advanced Architecture: Combine custom Convolutional layers for initial feature extraction with a pre-trained MobileNetV2 backbone.
4. Two-Stage Training:
   - Stage 1: Train the custom head and top layers while keeping the MobileNetV2 base frozen.
   - Stage 2: Fine-tune the top 100 layers of the base model with a reduced learning rate.
5. Automated Export & Zipping: Programmatically convert the trained model into TensorFlow Lite (.tflite) for mobile and TensorFlow.js (tfjs) for the web, bundle them with configuration JSONs and READMEs, and compress them into a single downloadable `.zip` file.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (alessiocorrado99/animals10)
Classes Used    : 
- farfalla (Butterfly)
- gallina (Hen)
- ragno (Spider)
- scoiattolo (Squirrel)
Input Shape     : Images are dynamically resized to (224, 224, 3) within the model architecture.

------------------------------------------------------------

MACHINE LEARNING PIPELINE ARCHITECTURE

1. Data Augmentation:
   - Built-in `Sequential` layer applying Random Flip, Rotation, Zoom, Contrast, and Translation.

2. Hybrid Model Architecture:
   Unlike standard transfer learning, this model introduces custom Conv2D layers *before* the pre-trained base:
   - Conv2D (32) + BatchNorm + MaxPool
   - Conv2D (64) + BatchNorm + MaxPool
   - Conv2D (3, 1x1 bottleneck) + Resizing(224, 224)
   - MobileNetV2 (Base Feature Extractor)
   - BatchNorm + Dense (256, L2 Regularization) + Dropout (0.5)
   - Dense (4 classes, Softmax)

------------------------------------------------------------

DEPLOYMENT ARTIFACTS & SUBMISSION STRUCTURE

The script is designed to act as an automated build tool. It creates a `submission/` folder and compresses it into a ready-to-deploy `.zip` archive containing:

submission/
    ├── tfjs_model/ (Web deployment artifacts)
    ├── tflite/ 
    │   ├── model.tflite (Mobile/Edge deployment)
    │   └── label.txt
    ├── saved_model/ (Native TF format)
    ├── klasifikasi-hewan-top4.keras
    ├── klasifikasi-hewan-top4.h5
    ├── training_config.json (Metadata containing accuracies and hyperparams)
    ├── README.md
    ├── requirements.txt
    └── notebook.ipynb

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install tensorflow kagglehub split-folders scikit-learn seaborn tensorflowjs

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/animals10-classification.git

2. Run the Python script or Notebook. The code will execute the entire pipeline from dataset downloading to ZIP file generation.
3. At the end of the execution, you will be prompted to upload an image to test the model's live inference capabilities.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Deep Learning & Computer Vision
- Hybrid CNN Architectures
- Automated Data Engineering
- TensorFlow.js & TensorFlow Lite Conversion
- MLOps Pipeline Automation

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Image Classification
- MobileNetV2 Transfer Learning
- Animals-10 Dataset
- MLOps Automation
- TFLite TFJS Deployment
- Deep Learning Portfolio
