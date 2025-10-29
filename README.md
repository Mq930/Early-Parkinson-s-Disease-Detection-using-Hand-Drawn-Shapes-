# SketchScreen - Early Parkinson's Detection Web Application

SketchScreen is a web-based application that uses machine learning to analyze hand-drawn spirals and waves for early detection of Parkinson's disease symptoms.

## Features

- Upload and analyze spiral and wave drawings
- Real-time image processing and analysis
- Detailed report generation with visualization
- Educational resources and support information
- Mobile-friendly interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, or Edge)
- 4GB RAM minimum (8GB recommended)

## Running the Application

1. Make sure your virtual environment is activated
2. Start the Flask server:
```bash
python app.py
```
3. Open your web browser and navigate to `http://localhost:5003`

## 🧠 Model Weights

The trained deep learning models for spiral and wave analysis are too large to be stored directly on GitHub.  
Please download them using the links below and place them in the `models/` directory before running the application.

### 📥 Download Links

- **Spiral Model:** [Download spiral_model weights](https://drive.google.com/file/d/1Q920jUU-ljhTGB631VixhhcSGMthtRRt/view?usp=drive_link)
- **Wave Model:** [Download wave_model weights](https://drive.google.com/file/d/1VTuTNjlXXyaJTYscCbIqYLKwNN62qa5p/view?usp=drive_link)


## Project Structure

```
sketchscreen/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── models/            # ML model files
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
└── utils/            # Utility functions and classes
```

1. Navigate to the self-test page
2. Upload your spiral and wave drawings
3. Fill in the required information
4. Submit for analysis
5. View and download your detailed report
