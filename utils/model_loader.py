import json
import os
from tensorflow import keras
import tensorflow as tf

class ModelLoader:
    def __init__(self):
        self.spiral_model = None
        self.wave_model = None
        # Spiral model has 4 conv layers: conv2d->conv2d_1->conv2d_2->conv2d_3 (32->64->128->256)
        self.spiral_last_conv = "conv2d_3"  # Last conv layer for spiral model (4th conv layer)
        # Wave model has 3 conv layers: conv2d->conv2d_1->conv2d_2 (32->64->128)
        self.wave_last_conv = "convo_3"     # Last conv layer for wave model (3rd conv layer)
        self.models_loaded = False

    def load_models(self):
        """Load both spiral and wave models."""
        if self.models_loaded:
            return True

        try:
            print("Loading models...")
            
            # Check if model files exist
            required_files = [
                'models/spiral_config.json',
                'models/wave_config.json',
                'models/spiral.weights.new.h5',  # Using new weight files
                'models/wave.weights.new.h5'     # Using new weight files
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required model file not found: {file_path}")

            # Load spiral model
            with open('models/spiral_config.json') as json_file:
                spiral_config = json_file.read()
            
            self.spiral_model = keras.models.model_from_json(spiral_config)
            if self.spiral_model is None:
                raise ValueError("Failed to create spiral model from config")
                
            self.spiral_model.load_weights('models/spiral.weights.new.h5')  # Using new weight file
            print("Spiral model loaded successfully")

            # Load wave model
            with open('models/wave_config.json') as json_file:
                wave_config = json_file.read()
            
            self.wave_model = keras.models.model_from_json(wave_config)
            if self.wave_model is None:
                raise ValueError("Failed to create wave model from config")
                
            self.wave_model.load_weights('models/wave.weights.new.h5')  # Using new weight file
            print("Wave model loaded successfully")

            # Configure optimizer
            optimizer_config = {
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-07,
                'amsgrad': False
            }
            optimizer = keras.optimizers.Adam(**optimizer_config)

            # Compile models
            self.spiral_model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.wave_model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Test predictions
            test_input = tf.zeros((1, 256, 256, 1))  # Test input for spiral model
            _ = self.spiral_model.predict(test_input)
            print("Spiral model prediction test successful")

            test_input = tf.zeros((1, 250, 550, 1))  # Test input for wave model
            _ = self.wave_model.predict(test_input)
            print("Wave model prediction test successful")

            self.models_loaded = True
            print("All models loaded and tested successfully")
            return True

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.spiral_model = None
            self.wave_model = None
            self.models_loaded = False
            return False

    def get_spiral_model(self):
        """Get the spiral analysis model."""
        if not self.models_loaded:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load models")
        return self.spiral_model

    def get_wave_model(self):
        """Get the wave analysis model."""
        if not self.models_loaded:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load models")
        return self.wave_model

    def get_last_conv_layer(self, is_wave=False):
        """Get the name of the last convolutional layer for Grad-CAM."""
        return self.wave_last_conv if is_wave else self.spiral_last_conv 