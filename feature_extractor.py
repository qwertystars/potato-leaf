from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load model once - use ResNet50 as it's more stable
effnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    # Load image as RGB - ResNet50 expects 224x224 by default
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    # Ensure we have 3 channels (RGB)
    if x.shape[-1] != 3:
        # Convert grayscale to RGB if needed
        if len(x.shape) == 2 or x.shape[-1] == 1:
            x = np.stack([x.squeeze()] * 3, axis=-1)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = effnet.predict(x, verbose=0)
    return feat.flatten()
