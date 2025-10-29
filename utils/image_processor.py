import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class ImageProcessor:
    def __init__(self):
        pass

    def process_spiral(self, img):
        """Process spiral drawing image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        gray = 255 - gray  # Invert colors
        return gray

    def process_wave(self, img):
        """Process wave drawing image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (550, 250))
        gray = 255 - gray  # Invert colors
        return gray

    def make_spiral_gradcam(self, img_array, model, last_conv_layer_name="conv2d_3", pred_index=None):
        """Generate Grad-CAM heatmap for spiral model.
        Spiral model architecture:
        - 4 conv layers (32->64->128->256 filters)
        - Using last conv layer (conv2d_3) with 256 filters
        """
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize using the spiral model's scale
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)
        return heatmap.numpy()

    def make_wave_gradcam(self, img_array, model, last_conv_layer_name="convo_3", pred_index=None):
        """Generate Grad-CAM heatmap for wave model using the original implementation."""
        grad_model = Model(
            inputs=model.input,  # Fixed input layer
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize and enhance contrast
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=None, is_wave=False):
        """Overlay heatmap on the original image."""
        if alpha is None:
            alpha = 0.7 if is_wave else 0.4  # Higher alpha for wave to make it more visible

        # Resize and enhance heatmap
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply Gaussian blur with smaller kernel for sharper edges
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        
        # Apply color mapping with enhanced contrast
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create the overlay with enhanced contrast
        superimposed_img = cv2.addWeighted(heatmap_color, alpha, image, 1 - alpha, 0)
        return superimposed_img

    def prepare_image_for_prediction(self, img, is_wave=False):
        """Prepare image for model prediction."""
        if is_wave:
            processed = self.process_wave(img)
            target_size = (550, 250)
        else:
            processed = self.process_spiral(img)
            target_size = (256, 256)
        
        processed = cv2.resize(processed, target_size)
        normalized = processed / 255.0
        input_img = np.expand_dims(normalized, axis=[0, -1])
        return input_img, processed 