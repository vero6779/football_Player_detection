# Football Player Detection ⚽

A production-ready YOLO11 object detection project built for Streamlit Cloud deployment. This application detects football players, goalkeepers, referees, and the ball in images using a model trained on a custom dataset from Roboflow.

## 🧠 Project Overview

This project provides a complete, modular structure to deploy a trained YOLO model using Streamlit. It uses the `ultralytics` YOLO API for inference and OpenCV for bounding box visualization. The application supports uploading an image or running inference on a default demo image, providing confidence score adjustments and interactive UI.

### Features
- Clean, modular code architecture suitable for GitHub portfolios
- Automatic fallback to CPU if GPU is unavailable
- Responsive, intuitive Streamlit UI with robust error handling and loading states
- Dynamic class name loading
- Local testing scripts

## 📁 Folder Structure

```
football-player-detection/
│
├── app.py                   # Local testing script
├── streamlit_app.py         # Main Streamlit web application
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (for Streamlit Cloud)
├── README.md                # Project documentation
│
├── model/
│   ├── best.pt              # YOLO11 trained weights
│   └── labels.txt           # Class names
│
├── utils/
│   ├── detector.py          # YOLOModel inference class
│   └── visualization.py     # Bounding box drawing functions
│
└── assets/
    └── demo.png             # Demo image
```

## 🚀 How to Run Locally

1. **Install Dependencies**
   It's highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Local Test Script**
   This will run inference on the demo image and save the output as `output.jpg`. It will print detections to the console.
   ```bash
   python app.py
   ```

3. **Run Streamlit App**
   Launch the web interface locally. It will open in your default browser.
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Deployment to Streamlit Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository, branch, and specify `streamlit_app.py` as the Main file path.
5. Click **Deploy!**

The included `packages.txt` ensures that `libgl1` (required by OpenCV) is automatically installed in the Streamlit Cloud Linux environment. `requirements.txt` includes `opencv-python-headless` for further compatibility in UI-less server environments.

## 📊 Output Example
The model outputs precise bounding boxes around detected entities, along with the class name and confidence score. The frontend provides a confidence slider to filter out less certain predictions dynamically. Original and annotated images are displayed side-by-side for comparison.
