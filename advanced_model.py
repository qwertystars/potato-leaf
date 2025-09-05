import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

# Set seaborn style for better visualizations
sns.set_style("whitegrid")
sns.set_palette("Set2")

class AdvancedLeafDiseaseModel:
    def __init__(self, model_type="densenet", input_shape=(224, 224, 3), num_classes=3):
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        
    def create_model(self):
        """Create advanced model with different architecture options - all accept 3-channel RGB"""
        # Ensure input shape is always 3-channel RGB
        input_shape = (224, 224, 3)

        if self.model_type == "densenet":
            self.base_model = DenseNet121(
                weights='imagenet',
                include_top=False, 
                input_shape=input_shape,
                pooling=None
            )
        elif self.model_type == "vgg16":
            self.base_model = VGG16(
                weights='imagenet', 
                include_top=False, 
                input_shape=input_shape,
                pooling=None
            )
        elif self.model_type == "resnet":
            self.base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape,
                pooling=None
            )
        else:
            # Default to ResNet50 if unknown model type
            self.base_model = ResNet50(
                weights='imagenet',
                include_top=False, 
                input_shape=input_shape,
                pooling=None
            )
        
        # Add custom classification head
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        
        # Freeze base model layers initially
        for layer in self.base_model.layers:
            layer.trainable = False

        # Unfreeze last few layers for fine-tuning
        if len(self.base_model.layers) > 20:
            for layer in self.base_model.layers[-20:]:
                layer.trainable = True

        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile model with optimized settings"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def preprocess_image(self, img_path):
        """Preprocess image to ensure 3-channel RGB format"""
        # Load image as RGB
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Ensure we have exactly 3 channels
        if img_array.shape[-1] == 1:
            # Convert grayscale to RGB
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.shape[-1] == 4:
            # Convert RGBA to RGB
            img_array = img_array[:, :, :3]

        # Normalize to [0, 1]
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def generate_heatmap(self, img_path, class_idx=None):
        """Generate Grad-CAM heatmap for disease localization"""
        # Load and preprocess original image for display
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Could not load image from {img_path}")

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Preprocess image for model
        img_array = self.preprocess_image(img_path)

        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.base_model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find convolutional layer for heatmap generation")

        # Create gradient model
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [last_conv_layer.output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        # Calculate gradients
        grads = tape.gradient(class_output, conv_outputs)
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

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        
        # Create superimposed image
        superimposed = heatmap_colored * 0.6 + original_img * 0.4
        superimposed = superimposed.astype(np.uint8)

        # Create visualization with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Heatmap only
        im = axes[0, 1].imshow(heatmap_resized, cmap='jet')
        axes[0, 1].set_title('Disease Attention Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Superimposed
        axes[1, 0].imshow(superimposed)
        axes[1, 0].set_title('Heatmap Overlay', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # Prediction results
        pred_result = self.predict_single(img_path)
        class_names = pred_result['class_names']
        probabilities = pred_result['all_probabilities']

        axes[1, 1].bar(range(len(class_names)), probabilities * 100)
        axes[1, 1].set_xlabel('Disease Classes')
        axes[1, 1].set_ylabel('Probability (%)')
        axes[1, 1].set_title('Classification Probabilities', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels([name.replace('_', ' ').title() for name in class_names],
                                   rotation=45, ha='right')

        # Highlight predicted class
        max_idx = np.argmax(probabilities)
        axes[1, 1].bar(max_idx, probabilities[max_idx] * 100, color='red', alpha=0.7)

        plt.tight_layout()

        return fig, heatmap_resized, superimposed, pred_result

    def predict_single(self, img_path):
        """Make prediction for a single image"""
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)

        # Standard class names for leaf diseases
        class_names = ['early_blight', 'healthy', 'late_blight']
        pred_class_idx = np.argmax(predictions[0])

        return {
            'predicted_class': class_names[pred_class_idx],
            'confidence': predictions[0][pred_class_idx],
            'all_probabilities': predictions[0],
            'class_names': class_names
        }

    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model

def create_data_generators(train_dir, validation_split=0.2, batch_size=32):
    """Create enhanced data generators with augmentation - ensures 3-channel RGB"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',  # Ensure RGB mode
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',  # Ensure RGB mode
        subset='validation'
    )
    
    return train_generator, validation_generator
