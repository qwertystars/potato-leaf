"""
Mobile Image Processor for Leaf Disease Analysis
Handles mobile camera issues and optimizes images for better disease detection
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

class MobileImageProcessor:
    def __init__(self):
        self.target_size = (224, 224)

    def preprocess_mobile_image(self, image_path, output_path=None):
        """
        Preprocess mobile camera image to handle green detection issues
        and optimize for disease analysis
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply mobile-specific preprocessing
            processed_image = self._apply_mobile_corrections(image_rgb)

            # Enhanced leaf segmentation
            processed_image = self._enhance_leaf_features(processed_image)

            # Noise reduction
            processed_image = self._reduce_noise(processed_image)

            # Save processed image if output path provided
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                return output_path

            return processed_image

        except Exception as e:
            print(f"Error in mobile image preprocessing: {e}")
            return None

    def _apply_mobile_corrections(self, image):
        """Apply corrections specific to mobile camera issues"""
        # Convert to LAB color space for better color correction
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])

        # Convert back to RGB
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return corrected

    def _enhance_leaf_features(self, image):
        """Enhance leaf features and reduce false green detection"""
        # Convert to HSV for better color space analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define multiple green ranges for better leaf detection
        # Lower green range
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])

        # Upper green range (for yellowish greens)
        lower_green2 = np.array([85, 40, 40])
        upper_green2 = np.array([95, 255, 255])

        # Create masks for green areas
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # Apply Gaussian blur to smooth edges
        green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)

        # Create 3-channel mask
        green_mask_3ch = cv2.merge([green_mask, green_mask, green_mask]) / 255.0

        # Enhance green areas while preserving disease symptoms
        enhanced = image.copy().astype(np.float32)

        # Slightly enhance green areas but preserve disease spots
        green_enhanced = enhanced * 1.1
        non_green_enhanced = enhanced * 0.95

        # Blend based on green mask
        result = green_enhanced * green_mask_3ch + non_green_enhanced * (1 - green_mask_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _reduce_noise(self, image):
        """Reduce noise while preserving important disease features"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        # Apply slight Gaussian blur to reduce camera artifacts
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)

        return denoised

    def detect_leaf_region(self, image_path):
        """Detect and extract the main leaf region from the image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to HSV for better segmentation
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

            # Create mask for green areas (leaf detection)
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([95, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Clean up the mask
            kernel = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image_rgb  # Return original if no leaf detected

            # Find the largest contour (main leaf)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_rgb.shape[1] - x, w + 2*padding)
            h = min(image_rgb.shape[0] - y, h + 2*padding)

            # Crop the leaf region
            leaf_region = image_rgb[y:y+h, x:x+w]

            return leaf_region

        except Exception as e:
            print(f"Error in leaf detection: {e}")
            return None

    def process_for_mobile_camera(self, image_path, enhance_for_analysis=True):
        """
        Complete processing pipeline for mobile camera images
        """
        try:
            # Step 1: Detect and extract leaf region
            leaf_region = self.detect_leaf_region(image_path)
            if leaf_region is None:
                # Fallback to full image processing
                return self.preprocess_mobile_image(image_path)

            # Step 2: Convert to PIL for additional enhancements
            pil_image = Image.fromarray(leaf_region)

            # Step 3: Enhance image quality for analysis
            if enhance_for_analysis:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.2)

                # Enhance color saturation slightly
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(1.1)

                # Sharpen the image slightly
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.1)

            # Step 4: Resize to target size while maintaining aspect ratio
            pil_image = self._resize_with_padding(pil_image, self.target_size)

            # Convert back to numpy array
            final_image = np.array(pil_image)

            return final_image

        except Exception as e:
            print(f"Error in mobile camera processing: {e}")
            return None

    def _resize_with_padding(self, image, target_size):
        """Resize image while maintaining aspect ratio using padding"""
        # Calculate aspect ratio
        aspect_ratio = image.width / image.height
        target_aspect = target_size[0] / target_size[1]

        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and white background
        result = Image.new('RGB', target_size, (255, 255, 255))

        # Calculate position to paste resized image (center)
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2

        # Paste resized image onto white background
        result.paste(resized, (x_offset, y_offset))

        return result

    def validate_leaf_image(self, image_path):
        """
        Validate if the image contains a valid leaf
        Returns: (is_valid, confidence_score, issues)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0.0, ["Could not read image"]

            issues = []
            confidence = 1.0

            # Check image size
            height, width = image.shape[:2]
            if width < 100 or height < 100:
                issues.append("Image too small")
                confidence *= 0.5

            # Check for green content (leaf detection)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([95, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            green_percentage = (np.sum(green_mask > 0) / (width * height)) * 100

            if green_percentage < 5:
                issues.append("Very little green content detected")
                confidence *= 0.7
            elif green_percentage < 15:
                issues.append("Low green content")
                confidence *= 0.8

            # Check blur level
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < 100:
                issues.append("Image appears blurry")
                confidence *= 0.6

            is_valid = confidence > 0.3 and len([i for i in issues if "too small" in i or "Could not read" in i]) == 0

            return is_valid, confidence, issues

        except Exception as e:
            return False, 0.0, [f"Validation error: {str(e)}"]

# Global processor instance
mobile_processor = MobileImageProcessor()

def process_mobile_upload(image_path, save_processed=True):
    """
    Process uploaded image from mobile device
    Returns: (processed_image_path, validation_info)
    """
    try:
        # Validate the image first
        is_valid, confidence, issues = mobile_processor.validate_leaf_image(image_path)

        validation_info = {
            "is_valid": is_valid,
            "confidence": round(confidence * 100, 1),
            "issues": issues
        }

        if not is_valid:
            return image_path, validation_info

        # Process the image
        if save_processed:
            # Create processed version filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_filename = f"processed_{base_name}.jpg"
            processed_path = os.path.join(os.path.dirname(image_path), processed_filename)

            # Process and save
            processed_image = mobile_processor.process_for_mobile_camera(image_path, enhance_for_analysis=True)

            if processed_image is not None:
                # Save processed image
                pil_processed = Image.fromarray(processed_image)
                pil_processed.save(processed_path, "JPEG", quality=95)
                return processed_path, validation_info
            else:
                return image_path, validation_info
        else:
            # Just return validation info
            return image_path, validation_info

    except Exception as e:
        print(f"Error in mobile upload processing: {e}")
        validation_info = {
            "is_valid": False,
            "confidence": 0.0,
            "issues": [f"Processing error: {str(e)}"]
        }
        return image_path, validation_info
