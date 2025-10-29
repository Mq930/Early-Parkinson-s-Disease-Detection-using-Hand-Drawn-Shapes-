import os
import cv2
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

class ReportGenerator:
    def __init__(self, image_processor, model_loader):
        self.image_processor = image_processor
        self.model_loader = model_loader
        
    def _image_to_base64(self, img):
        """Convert an image to base64 string."""
        if isinstance(img, np.ndarray):
            # Convert OpenCV image to PIL Image
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:  # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate_report(self, spiral_img, wave_img, user_info=None):
        """Generate analysis report in HTML format."""
        # Process images and get predictions
        spiral_input, spiral_processed = self.image_processor.prepare_image_for_prediction(spiral_img, is_wave=False)
        wave_input, wave_processed = self.image_processor.prepare_image_for_prediction(wave_img, is_wave=True)

        # Get models
        spiral_model = self.model_loader.get_spiral_model()
        wave_model = self.model_loader.get_wave_model()

        # Get predictions
        spiral_pred = spiral_model.predict(spiral_input)[0][0]
        wave_pred = wave_model.predict(wave_input)[0][0]

        # Generate Grad-CAM heatmaps
        spiral_heatmap = self.image_processor.make_spiral_gradcam(
            spiral_input, spiral_model, self.model_loader.get_last_conv_layer(is_wave=False))
        wave_heatmap = self.image_processor.make_wave_gradcam(
            wave_input, wave_model, self.model_loader.get_last_conv_layer(is_wave=True))

        # Create heatmap overlays
        spiral_overlay = self.image_processor.overlay_heatmap(spiral_heatmap, spiral_processed, is_wave=False)
        wave_overlay = self.image_processor.overlay_heatmap(wave_heatmap, wave_processed, is_wave=True)

        # Convert images to base64
        spiral_b64 = self._image_to_base64(spiral_processed)
        wave_b64 = self._image_to_base64(wave_processed)
        spiral_overlay_b64 = self._image_to_base64(spiral_overlay)
        wave_overlay_b64 = self._image_to_base64(wave_overlay)

        # Determine overall result and confidence
        spiral_has_parkinsons = spiral_pred > 0.5
        wave_has_parkinsons = wave_pred > 0.5
        
        # New confidence level logic
        if spiral_pred <= 0.5:
            confidence_level = spiral_pred
            confidence_source = 'Negative'
            has_parkinsons = False  # Override if confidence is low
        elif wave_pred <= 0.5:
            confidence_level = wave_pred
            confidence_source = 'Negative'
            has_parkinsons = False  # Override if confidence is low
        else:
            confidence_level = max(spiral_pred, wave_pred)
            confidence_source = 'Positive'
            has_parkinsons = True

        # Generate result message and next steps based on confidence
        if confidence_level <= 0.5:
            result_message = "No significant indicators of Parkinson's disease detected"
            next_steps = "Continue with regular health check-ups and maintain a healthy lifestyle. If you have any concerns, consult with your healthcare provider during your next routine visit."
            gradcam_desc = "The analysis shows minimal to no areas of concern in your drawings. The heatmap highlights are within normal ranges."
        else:
            result_message = "Our analysis indicates potential early signs of Parkinson's disease"
            next_steps = "We recommend consulting a neurologist within the next 2-4 weeks for a professional evaluation."
            gradcam_desc = """The heatmap analysis of your drawings shows areas of potential concern:

• Red regions indicate significant inconsistencies in drawing patterns

• Yellow areas show moderate variations from expected patterns

• Blue areas represent slight deviations from typical drawing characteristics

These highlighted regions are analyzed by our AI model to detect early indicators of Parkinson's disease."""

        # Parse user info
        name = user_info.get('name', '') if user_info else ''
        age = user_info.get('age', '') if user_info else ''
        gender = user_info.get('gender', '') if user_info else ''

        # Prepare PDF filename for JS
        safe_name = str(name).replace(' ', '_')
        pdf_filename = f'Parkinsons_Report_{safe_name}_{datetime.now().strftime("%Y%m%d")}.pdf'

        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parkinson's Early Detection Report</title>
            <script src='https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js'></script>
            <style>
                @page {{ size: A4; margin: 0; }}
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: white;
                    color: #333;
                    display: flex;
                    justify-content: center;
                }}
                .report-container {{
                    max-width: 165mm;  /* Fine-tuned from 170mm */
                    margin: 0 auto;
                    padding: 15mm;  /* Reduced from 20mm */
                    background: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    font-size: 11px;
                }}
                h1, h2 {{
                    color: #2c3e50;
                    margin: 0 0 15px 0;
                }}
                .report-header {{
                    text-align: center;
                    border-bottom: 2px solid #27ae60;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .report-header h1 {{
                    font-size: 24px;  /* Reduced from 28px */
                }}
                .date {{
                    color: #666;
                    font-size: 14px;  /* Reduced from 16px */
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 8px;
                    background: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .patient-info {{
                    background: #f8f9fa;
                }}
                .result-section {{
                    background: #e8f5e9;
                    border-left: 4px solid #27ae60;
                }}
                .confidence-box {{
                    margin: 15px 0;
                    padding: 10px;
                    background: rgba(255,255,255,0.7);
                    border-radius: 4px;
                }}
                .images-section {{
                    text-align: center;
                }}
                .image-container {{
                    display: flex;
                    justify-content: space-between;
                    gap: 20px;
                    margin-bottom: 20px;
                    width: 100%;
                }}
                .image-block {{
                    flex: 1;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .image-block h3 {{
                    margin: 0 0 15px 0;
                    font-size: 16px;
                    color: #2c3e50;
                }}
                .image-block img {{
                    width: 100%;
                    height: 150px;  /* Reduced from 180px */
                    object-fit: contain;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                }}
                .resources-list {{
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }}
                .resources-list li {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }}
                .resource-link {{
                    color: #27ae60;
                    text-decoration: none;
                    font-weight: bold;
                    font-size: 16px;
                }}
                .resource-desc {{
                    display: block;
                    color: #666;
                    margin-top: 8px;
                    line-height: 1.4;
                }}
                .disclaimer {{
                    background: #fff3e0;
                    font-size: 14px;
                    line-height: 1.6;
                }}
                .gradcam-description {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    line-height: 1.8;
                }}
                .gradcam-description p {{
                    margin: 0;
                }}
                @media print {{
                    body {{ 
                        margin: 0;
                        padding: 0;
                        background: white;
                        display: flex;
                        justify-content: center;
                    }}
                    .report-container {{ 
                        width: 155mm;  /* Fine-tuned from 160mm */
                        margin: 0 auto;
                        padding: 12mm;  /* Reduced from 15mm */
                        box-shadow: none;
                    }}
                    .section {{
                        break-inside: avoid;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="report-container">
                <div class="report-header">
                    <h1>Parkinson's Early Detection Report</h1>
                    <div class="date">Date: {datetime.now().strftime('%B %d, %Y')}</div>
                </div>

                <div class="section patient-info">
                    <h2>Patient Information</h2>
                    <div style="margin: 10px 0;"><strong>Name:</strong> {name}</div>
                    <div style="margin: 10px 0;"><strong>Age:</strong> {age}</div>
                    <div style="margin: 10px 0;"><strong>Gender:</strong> {gender}</div>
                </div>

                <div class="section result-section">
                    <h2>Analysis Result</h2>
                    <div style="margin: 15px 0;"><strong>{result_message}</strong></div>
                    <div class="confidence-box">
                        <strong>Confidence Level:</strong> {confidence_level:.2%} <span style="color:#666;">({confidence_source})</span>
                    </div>
                    <div style="margin: 15px 0;"><strong>Recommended Action:</strong> {next_steps}</div>
                </div>

                <div class="section images-section">
                    <h2>Drawing Analysis</h2>
                    <div style="margin-bottom: 20px;" class="gradcam-description">
                        <p style="white-space: pre-line;">{gradcam_desc}</p>
                    </div>
                    <div class="image-row">
                        <div class="image-container">
                            <div class="image-block">
                                <h3>Spiral Drawing - Original</h3>
                                <img src="data:image/png;base64,{spiral_b64}" alt="Original Spiral">
                            </div>
                            <div class="image-block">
                                <h3>Spiral Drawing - Analysis</h3>
                                <img src="data:image/png;base64,{spiral_overlay_b64}" alt="Spiral Analysis">
                            </div>
                        </div>
                    </div>
                    <div class="image-row">
                        <div class="image-container">
                            <div class="image-block">
                                <h3>Wave Drawing - Original</h3>
                                <img src="data:image/png;base64,{wave_b64}" alt="Original Wave">
                            </div>
                            <div class="image-block">
                                <h3>Wave Drawing - Analysis</h3>
                                <img src="data:image/png;base64,{wave_overlay_b64}" alt="Wave Analysis">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section resources">
                    <h2>Educational Resources</h2>
                    <ul class="resources-list">
                        <li>
                            <a class="resource-link" href="https://www.parkinson.org/" target="_blank">Parkinson's Foundation</a>
                            <span class="resource-desc">Comprehensive information, support, and resources for people with Parkinson's and their families.</span>
                        </li>
                        <li>
                            <a class="resource-link" href="https://www.michaeljfox.org/" target="_blank">Michael J. Fox Foundation</a>
                            <span class="resource-desc">Leading research, news, and community support for Parkinson's disease.</span>
                        </li>
                        <li>
                            <a class="resource-link" href="https://www.pdf.org/" target="_blank">Parkinson's Disease Foundation</a>
                            <span class="resource-desc">Educational materials, research updates, and support programs for Parkinson's patients.</span>
                        </li>
                    </ul>
                </div>

                <div class="section disclaimer">
                    <h2>Medical Disclaimer</h2>
                    <p style="margin: 10px 0;">This report is for informational purposes only and does not constitute a formal medical diagnosis. The analysis is based on computer vision algorithms and should not be used as a substitute for professional medical evaluation.</p>
                    <p style="margin: 10px 0;">If you have concerns about Parkinson's disease or any other medical condition, please consult with a qualified healthcare provider.</p>
                </div>
            </div>

            <script>
                function downloadPDF() {{
                    const element = document.querySelector('.report-container');
                    const opt = {{
                        filename: '{pdf_filename}',
                        margin: [5, 8, -5, -85],  /* Fine-tuned margins [top, right, bottom, left] */
                        image: {{ type: 'jpeg', quality: 1.00 }},
                        html2canvas: {{ 
                            scale: 1.45,  /* Fine-tuned from 1.5 */
                            useCORS: true,
                            letterRendering: true,
                            scrollY: 0,
                            windowWidth: element.offsetWidth,
                            windowHeight: element.offsetHeight
                        }},
                        jsPDF: {{ 
                            unit: 'mm',
                            format: 'a4',
                            orientation: 'portrait',
                            hotfixes: ["px_scaling"],
                            compress: true
                        }},
                        pagebreak: {{ mode: ['avoid-all', 'css', 'legacy'] }}
                    }};
                    html2pdf().set(opt).from(element).save();
                }}
            </script>
            <div style="text-align:center;margin:20px;">
                <button onclick="downloadPDF()" style="
                    background: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;">
                    Download as PDF
                </button>
            </div>
        </body>
        </html>
        """
        return report_html 
        return report_html 