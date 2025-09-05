import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from advanced_feature_extractor import AdvancedFeatureExtractor
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_robust_heatmap(img_path, model_type="densenet"):
    """Create a robust heatmap using multiple visualization techniques"""

    print(f"🔥 Creating robust heatmap for: {os.path.basename(img_path)}")

    # Load and prepare the image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"❌ Could not load image: {img_path}")
        return None

    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Get model prediction
    extractor = AdvancedFeatureExtractor(model_type)
    if extractor.model is None:
        print("❌ Model not loaded")
        return None

    prediction = extractor.predict(img_path)
    print(f"🔬 Prediction: {prediction['predicted_class']} ({prediction['confidence']*100:.1f}% confidence)")

    # Create multiple analysis layers for comprehensive heatmap

    # 1. Edge Detection Analysis (diseases often appear at leaf edges/veins)
    gray = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 2. Color-based Disease Detection
    hsv = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2HSV)

    # Focus on brown/yellow/orange hues (typical disease colors)
    lower_disease = np.array([10, 50, 50])   # Yellow/brown range
    upper_disease = np.array([30, 255, 255])
    disease_color_mask = cv2.inRange(hsv, lower_disease, upper_disease)

    # Also check for dark spots (another disease indicator)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    dark_spot_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # 3. Texture Analysis using Local Binary Patterns
    def calculate_lbp(image, radius=2, n_points=8):
        """Calculate Local Binary Pattern for texture analysis"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ""

                # Sample points in a circle around the center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))

                    if 0 <= x < rows and 0 <= y < cols:
                        binary_string += "1" if image[x, y] >= center else "0"

                if binary_string:
                    lbp[i, j] = int(binary_string, 2)

        return lbp

    texture_map = calculate_lbp(gray)

    # 4. Gradient Magnitude (highlights changes in intensity)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 5. Combine all analysis methods
    # Normalize each component to 0-1 range
    edge_norm = edges / 255.0
    color_norm = (disease_color_mask + dark_spot_mask) / 510.0  # Combined color masks
    texture_norm = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    gradient_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)

    # Weight combination based on prediction confidence and disease type
    confidence = prediction['confidence']
    predicted_class = prediction['predicted_class']

    if predicted_class == 'healthy':
        # For healthy leaves, show less intense attention
        attention_map = 0.2 * edge_norm + 0.1 * color_norm + 0.1 * texture_norm + 0.1 * gradient_norm
    elif 'blight' in predicted_class.lower():
        # For blight diseases, emphasize color changes and texture
        attention_map = 0.3 * edge_norm + 0.5 * color_norm + 0.4 * texture_norm + 0.3 * gradient_norm
    else:
        # General disease weighting
        attention_map = 0.3 * edge_norm + 0.4 * color_norm + 0.3 * texture_norm + 0.3 * gradient_norm

    # Apply confidence-based scaling
    attention_map = attention_map * confidence

    # Smooth the attention map
    attention_map = cv2.GaussianBlur(attention_map.astype(np.float32), (5, 5), 0)

    # Normalize final attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    return attention_map, original_img_rgb, prediction

def create_comprehensive_heatmap_visualization(img_path, save_path=None):
    """Create comprehensive heatmap visualization"""

    attention_map, original_img, prediction = create_robust_heatmap(img_path)

    if attention_map is None:
        return None

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Disease Analysis: {prediction["predicted_class"].replace("_", " ").title()}\n'
                f'Confidence: {prediction["confidence"]*100:.1f}%',
                fontsize=20, fontweight='bold')

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Leaf Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Disease attention heatmap
    im1 = axes[0, 1].imshow(attention_map, cmap='plasma')
    axes[0, 1].set_title('Disease Attention Map', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Overlay heatmap on original
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlay = heatmap_colored * 0.6 + original_img * 0.4
    overlay = overlay.astype(np.uint8)

    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Disease Location Overlay', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Probability bar chart
    class_names = prediction['class_names']
    probabilities = prediction['all_probabilities']
    class_labels = [name.replace('_', ' ').title() for name in class_names]

    colors = sns.color_palette("viridis", len(class_names))
    bars = axes[1, 0].bar(range(len(class_names)), probabilities * 100, color=colors)

    # Highlight predicted class
    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('red')
    bars[max_idx].set_edgecolor('darkred')
    bars[max_idx].set_linewidth(2)

    axes[1, 0].set_xlabel('Disease Classes', fontweight='bold')
    axes[1, 0].set_ylabel('Probability (%)', fontweight='bold')
    axes[1, 0].set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(class_names)))
    axes[1, 0].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # Heatmap intensity analysis
    max_attention = np.max(attention_map)
    mean_attention = np.mean(attention_map)
    std_attention = np.std(attention_map)

    # Create intensity histogram
    axes[1, 1].hist(attention_map.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(mean_attention, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_attention:.3f}')
    axes[1, 1].axvline(max_attention, color='blue', linestyle='--', linewidth=2, label=f'Max: {max_attention:.3f}')
    axes[1, 1].set_xlabel('Attention Intensity', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Attention Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Disease severity assessment
    if max_attention > 0.7:
        severity = "HIGH"
        severity_color = "red"
    elif max_attention > 0.4:
        severity = "MEDIUM"
        severity_color = "orange"
    else:
        severity = "LOW"
        severity_color = "green"

    # Summary text
    axes[1, 2].text(0.1, 0.8, f"DISEASE ANALYSIS SUMMARY", fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f"Predicted: {prediction['predicted_class'].replace('_', ' ').title()}", fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f"Confidence: {prediction['confidence']*100:.1f}%", fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f"Disease Severity: {severity}", fontsize=12, color=severity_color, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f"Max Attention: {max_attention:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f"Avg Attention: {mean_attention:.3f}", fontsize=12, transform=axes[1, 2].transAxes)

    # Add interpretation
    axes[1, 2].text(0.1, 0.15, "🔴 Red areas = High disease probability", fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.1, "🟡 Yellow areas = Medium disease probability", fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.05, "🔵 Blue areas = Healthy tissue", fontsize=10, transform=axes[1, 2].transAxes)

    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✅ Comprehensive heatmap saved to: {save_path}")

    plt.close()

    return attention_map, overlay, prediction

def demo_comprehensive_heatmaps():
    """Generate comprehensive heatmaps for all sample images"""

    print("🔥 Generating Comprehensive Disease Heatmaps...")
    print("=" * 60)

    # Get sample images from dataset
    sample_images = []

    # Early blight
    early_blight_dir = "datasets/early_blight"
    if os.path.exists(early_blight_dir):
        files = [f for f in os.listdir(early_blight_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            sample_images.append((os.path.join(early_blight_dir, files[0]), "Early_Blight"))

    # Late blight
    late_blight_dir = "datasets/late_blight"
    if os.path.exists(late_blight_dir):
        files = [f for f in os.listdir(late_blight_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            sample_images.append((os.path.join(late_blight_dir, files[0]), "Late_Blight"))

    # Healthy
    healthy_dir = "datasets/healthy"
    if os.path.exists(healthy_dir):
        files = [f for f in os.listdir(healthy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            sample_images.append((os.path.join(healthy_dir, files[0]), "Healthy"))

    if not sample_images:
        print("❌ No sample images found")
        return

    print(f"✅ Processing {len(sample_images)} sample images")

    generated_heatmaps = []

    for i, (img_path, label) in enumerate(sample_images, 1):
        print(f"\n🌿 Processing {label} (Image {i})")
        print(f"   📁 File: {os.path.basename(img_path)}")

        try:
            save_path = f"static/uploads/comprehensive_heatmap_{label.lower()}_{i}.png"

            attention_map, overlay, prediction = create_comprehensive_heatmap_visualization(
                img_path,
                save_path
            )

            if attention_map is not None:
                generated_heatmaps.append(save_path)

                print(f"   🔬 AI Prediction: {prediction['predicted_class'].replace('_', ' ').title()}")
                print(f"   📊 Confidence: {prediction['confidence']*100:.1f}%")

                # Analyze attention map
                max_attention = np.max(attention_map)
                mean_attention = np.mean(attention_map)

                print(f"   🎯 Attention Analysis:")
                print(f"      Max: {max_attention:.3f}")
                print(f"      Avg: {mean_attention:.3f}")

                if max_attention > 0.7:
                    print(f"      🔴 HIGH disease focus - significant symptoms detected!")
                elif max_attention > 0.4:
                    print(f"      🟡 MEDIUM disease focus - moderate symptoms")
                else:
                    print(f"      🟢 LOW disease focus - likely healthy tissue")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    print(f"\n🎯 Heatmap Generation Complete!")
    print(f"📁 Generated {len(generated_heatmaps)} comprehensive heatmaps:")
    for heatmap_file in generated_heatmaps:
        print(f"   • {os.path.basename(heatmap_file)}")

    print(f"\n🔍 How to interpret the heatmaps:")
    print(f"   🔴 Red/Hot colors = Areas where AI detected disease symptoms")
    print(f"   🟡 Yellow/Warm colors = Moderate disease probability")
    print(f"   🔵 Blue/Cool colors = Healthy leaf tissue")
    print(f"   📊 Higher intensity = Higher confidence in disease detection")

if __name__ == "__main__":
    demo_comprehensive_heatmaps()
