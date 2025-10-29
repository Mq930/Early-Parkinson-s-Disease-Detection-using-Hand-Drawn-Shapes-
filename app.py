from flask import Flask, render_template, request, jsonify, send_file, url_for
from pathlib import Path
import cv2
import numpy as np
import os
from utils.image_processor import ImageProcessor
from utils.model_loader import ModelLoader
from utils.report_generator import ReportGenerator
import sys
import json

# Create required directories
Path("static/css").mkdir(parents=True, exist_ok=True)
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("static/reports").mkdir(parents=True, exist_ok=True)
Path("utils").mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_url_path='/static')

# Initialize model loader and image processor
print("Initializing model loader and image processor...")
model_loader = ModelLoader()
if not model_loader.load_models():
    print("Error: Failed to load models at startup", file=sys.stderr)
    sys.exit(1)
print("Models loaded successfully")

image_processor = ImageProcessor()

# Routes for HTML pages
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/self-test')
def self_test():
    return render_template('self_test.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get uploaded files
        spiral_file = request.files.get('spiral')
        wave_file = request.files.get('wave')
        user_info_raw = request.form.get('user_info')

        if not spiral_file or not wave_file:
            return jsonify({
                'status': 'error',
                'message': 'Both spiral and wave drawings are required'
            }), 400

        # Parse and validate user info
        try:
            user_info = json.loads(user_info_raw)
            name = user_info.get('name', '').strip()
            age = int(user_info.get('age', 0))
            gender = user_info.get('gender', '').strip()
            if not name or not (18 <= age <= 60) or gender not in ['Male', 'Female', 'Other']:
                raise ValueError
        except Exception:
            return jsonify({
                'status': 'error',
                'message': 'Invalid user information. Please provide name, age (18-60), and gender.'
            }), 400

        # Save uploaded files
        spiral_path = os.path.join('static/uploads', 'spiral.png')
        wave_path = os.path.join('static/uploads', 'wave.png')
        
        spiral_file.save(spiral_path)
        wave_file.save(wave_path)

        # Read images
        spiral_img = cv2.imread(spiral_path)
        wave_img = cv2.imread(wave_path)

        if spiral_img is None or wave_img is None:
            return jsonify({
                'status': 'error',
                'message': 'Error reading uploaded images'
            }), 400

        # Generate report
        report_generator = ReportGenerator(image_processor=image_processor, model_loader=model_loader)
        report_html = report_generator.generate_report(spiral_img, wave_img, user_info)

        # Save report
        report_path = os.path.join('static/reports', 'report.html')
        with open(report_path, 'w') as f:
            f.write(report_html)

        return jsonify({
            'status': 'success',
            'message': 'Analysis complete',
            'report_url': url_for('static', filename='reports/report.html')
        })

    except Exception as e:
        print(f"Error in analyze route: {str(e)}", file=sys.stderr)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Form submission endpoints
@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    try:
        data = request.form
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        # Here you would typically send an email or store in database
        # For now, we'll just return success
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your message. We will get back to you soon!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/subscribe-newsletter', methods=['POST'])
def subscribe_newsletter():
    try:
        email = request.form.get('email')
        if not email:
            return jsonify({
                'status': 'error',
                'message': 'Email is required'
            }), 400
            
        # Here you would typically add the email to your newsletter list
        # For now, we'll just return success
        return jsonify({
            'status': 'success',
            'message': 'Thank you for subscribing to our newsletter!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download-guide/<guide_type>')
def download_guide(guide_type):
    try:
        # Here you would typically serve the actual guide file
        # For now, we'll just return a success message
        return jsonify({
            'status': 'success',
            'message': f'Download started for {guide_type} guide'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)