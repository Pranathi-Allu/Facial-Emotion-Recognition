Facial Emotion Recognition (FER) — Setup & Implementation Guide

Dataset: Face expression recognition dataset on Kaggle
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

Files:
- FER.ipynb — Jupyter Notebook for training a model to detect 7 standard emotions
- FER.keras — Pre-trained emotion detection model
- Main.py — Final Python script to run the emotion detection using webcam or with input video

Step-by-Step Implementation

1. Create a Virtual Environment
   - Open Visual Studio Code.
   - Press Ctrl + Shift + P to open the command palette.
   - Search and select "Python: Create Environment".
   - Choose "Venv" as the environment type.
   - Select a Python interpreter (e.g., Python 3.12 or any version you have installed).
   - Name the environment .venv or any preferred name.

2. Activate the Virtual Environment
   - On Windows, open the terminal and type:
     .venv\Scripts\activate
   - On macOS or Linux, use:
     source .venv/bin/activate

3. Install Required Dependencies
   - With the virtual environment activated, install the necessary packages using the following command:
     pip install opencv-python mediapipe tensorflow numpy scipy

4. Project Setup
   - Place the following files in your working directory:
     - FERMain.py
     - FER.keras (the trained model file)
   - Ensure the model path in FERMain.py correctly points to FER.keras

5. Run the Project
   - In the terminal (with the virtual environment still activated), run the script:
     python FERMain.py
   - This should launch your webcam or process the input video and start detecting facial emotions in real time.
