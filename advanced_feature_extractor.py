import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from advanced_model import AdvancedLeafDiseaseModel

# Set seaborn style for better visualizations
sns.set_style("whitegrid")
sns.set_palette("viridis")

class AdvancedFeatureExtractor:
    def __init__(self, model_type="densenet"):
        self.model_type = model_type
        self.model = None
        self.class_info = None
        self.load_model()
    
    def load_model(self):
        """Load the advanced model and class information"""
        model_path = f"models/{self.model_type}_leaf_disease_model.h5"
        class_info_path = f"models/{self.model_type}_class_info.pkl"
        
        if os.path.exists(model_path) and os.path.exists(class_info_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.class_info = joblib.load(class_info_path)
                print(f"✅ Loaded {self.model_type} model successfully")
                return True
            except Exception as e:
                print(f"❌ Error loading {self.model_type} model: {e}")
                return False
        else:
            print(f"❌ {self.model_type} model not found. Please train it first.")
            return False
    
    def preprocess_image(self, img_path):
        """Preprocess image to ensure 3-channel RGB format"""
        # Load image as RGB with target size
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)

        # Ensure we have exactly 3 channels (RGB)
        if x.shape[-1] == 1:
            # Convert grayscale to RGB by repeating the single channel
            x = np.repeat(x, 3, axis=-1)
        elif x.shape[-1] == 4:
            # Convert RGBA to RGB by dropping alpha channel
            x = x[:, :, :3]

        # Normalize to [0,1] and add batch dimension
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        return x
    
    def predict(self, img_path):
        """Make prediction with probability scores"""
        if self.model is None:
            raise Exception("Model not loaded. Please train the model first.")
        
        x = self.preprocess_image(img_path)
        predictions = self.model.predict(x, verbose=0)
        
        # Get class names from class_info or use defaults
        if self.class_info and 'class_indices' in self.class_info:
            class_names = list(self.class_info['class_indices'].keys())
        else:
            class_names = ['early_blight', 'healthy', 'late_blight']

        # Get predicted class
        pred_class_idx = np.argmax(predictions[0])
        pred_class = class_names[pred_class_idx]
        confidence = predictions[0][pred_class_idx]
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': predictions[0],
            'class_names': class_names
        }
    
    def generate_heatmap(self, img_path, save_path=None):
        """Generate Grad-CAM heatmap for disease localization"""
        if self.model is None:
            raise Exception("Model not loaded. Please train the model first.")
        
        # Load and preprocess original image for display
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Could not load image from {img_path}")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model prediction
        img_array = self.preprocess_image(img_path)
        
        # Find the last convolutional layer - improved search
        last_conv_layer = None

        # First, try to find conv layers in the main model
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
            # If it's a functional model, check if it has layers attribute
            elif hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if hasattr(sublayer, 'output_shape') and len(sublayer.output_shape) == 4:
                        last_conv_layer = sublayer
                        break
                if last_conv_layer:
                    break

        # If still not found, search by layer type name
        if last_conv_layer is None:
            for layer in reversed(self.model.layers):
                layer_name = layer.__class__.__name__.lower()
                if 'conv' in layer_name or 'batch' in layer_name:
                    # Get the actual conv layer from batch norm layer's input
                    if hasattr(layer, 'input_spec'):
                        last_conv_layer = layer
                        break

        # Final fallback - use GlobalAveragePooling2D input
        if last_conv_layer is None:
            for layer in self.model.layers:
                if layer.__class__.__name__ == 'GlobalAveragePooling2D':
                    # The layer before GAP should be conv
                    layer_idx = self.model.layers.index(layer)
                    if layer_idx > 0:
                        last_conv_layer = self.model.layers[layer_idx - 1]
                        break

        if last_conv_layer is None:
            # Create a simple attention map based on the final dense layer gradients
            print("⚠️ No convolutional layer found. Creating simplified attention map...")
            return self._create_simplified_heatmap(img_path, save_path)

        print(f"🔍 Using layer: {last_conv_layer.name} ({last_conv_layer.__class__.__name__})")

        # Create gradient model
        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
        except Exception as e:
            print(f"❌ Error creating gradient model: {e}")
            return self._create_simplified_heatmap(img_path, save_path)

        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        grads = tape.gradient(class_output, conv_outputs)

        if grads is None:
            print("⚠️ Could not compute gradients. Creating simplified attention map...")
            return self._create_simplified_heatmap(img_path, save_path)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        # Normalize heatmap
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        return self._create_heatmap_visualization(original_img, heatmap_resized, img_path, save_path)

    def _create_simplified_heatmap(self, img_path, save_path=None):
        """Create a simplified attention map when Grad-CAM fails"""
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Create a simple attention map based on image gradients and prediction confidence
        img_array = self.preprocess_image(img_path)
        prediction = self.predict(img_path)

        # Create a basic attention map using image gradients
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        # Use Sobel gradients to highlight edges (disease often appears at edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine gradients
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize to 0-1
        heatmap = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

        # Enhance based on prediction confidence
        confidence = prediction['confidence']
        if confidence < 0.5:  # Low confidence, show more diffuse attention
            heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (15, 15), 0)

        return self._create_heatmap_visualization(original_img, heatmap, img_path, save_path)

    def _create_heatmap_visualization(self, original_img, heatmap, img_path, save_path=None):
        """Create the final heatmap visualization with seaborn styling"""
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )

        # Create superimposed image
        superimposed = heatmap_colored * 0.6 + original_img * 0.4
        superimposed = superimposed.astype(np.uint8)
        
        # Create visualization with multiple views using seaborn styling
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Heatmap only with seaborn colormap
        im = axes[0, 1].imshow(heatmap, cmap='plasma')
        axes[0, 1].set_title('Disease Attention Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Superimposed
        axes[1, 0].imshow(superimposed)
        axes[1, 0].set_title('Heatmap Overlay', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Prediction results with seaborn styling
        pred_result = self.predict(img_path)
        class_names = pred_result['class_names']
        probabilities = pred_result['all_probabilities']
        
        # Use seaborn barplot for better styling
        class_labels = [name.replace('_', ' ').title() for name in class_names]
        colors = sns.color_palette("viridis", len(class_names))

        bars = axes[1, 1].bar(range(len(class_names)), probabilities * 100, color=colors)
        axes[1, 1].set_xlabel('Disease Classes', fontweight='bold')
        axes[1, 1].set_ylabel('Probability (%)', fontweight='bold')
        axes[1, 1].set_title('Classification Probabilities', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_labels, rotation=45, ha='right')

        # Highlight predicted class with accent color
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color(sns.color_palette("Reds_r")[2])
        bars[max_idx].set_edgecolor('darkred')
        bars[max_idx].set_linewidth(2)

        # Add grid for better readability
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"✅ Heatmap saved to: {save_path}")

        plt.close()

        return heatmap, superimposed, pred_result

# Backward compatibility functions
def extract_features(img_path, model_type="densenet"):
    """Extract features using advanced model (for backward compatibility)"""
    extractor = AdvancedFeatureExtractor(model_type)
    if extractor.model is None:
        # Fallback to original ResNet50 if advanced model not available
        from feature_extractor import extract_features as original_extract
        return original_extract(img_path)
    
    # Extract features from dense layer before final classification
    x = extractor.preprocess_image(img_path)
    
    # Get features from second-to-last dense layer
    try:
        feature_model = tf.keras.models.Model(
            inputs=extractor.model.input,
            outputs=extractor.model.get_layer('dense_2').output
        )
        features = feature_model.predict(x, verbose=0)
        return features.flatten()
    except:
        # If dense_2 layer not found, use global average pooling output
        for layer in extractor.model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                feature_model = tf.keras.models.Model(
                    inputs=extractor.model.input,
                    outputs=layer.output
                )
                features = feature_model.predict(x, verbose=0)
                return features.flatten()

        # Final fallback - use model prediction as features
        features = extractor.model.predict(x, verbose=0)
        return features.flatten()

def predict_with_heatmap(img_path, model_type="densenet", save_heatmap=True):
    """Predict disease and generate heatmap"""
    extractor = AdvancedFeatureExtractor(model_type)
    
    if extractor.model is None:
        return None, None
    
    # Generate heatmap and prediction
    save_path = None
    if save_heatmap:
        filename = os.path.basename(img_path)
        # Save to static/results so the template path works consistently
        save_path = f"static/results/heatmap_{filename}.png"

    try:
        heatmap, overlay, prediction = extractor.generate_heatmap(img_path, save_path)
        return prediction, save_path
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        # Fallback to prediction only
        try:
            prediction = extractor.predict(img_path)
            return prediction, None
        except:
            return None, None
