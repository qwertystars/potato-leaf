"""
Camera Image Service
Handles camera image processing, validation, and storage.
"""

import os
import io
import uuid
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from PIL import Image, ExifTags
import cv2
import numpy as np
import json

from ..models.camera_capture import (
    CapturedImage, ProcessingStep, CameraSettings, QualityCheckResult,
    ProcessingStatus, StepStatus, storage
)


class CameraImageService:
    """Service for processing camera-captured images"""
    
    def __init__(self, upload_dir: str = "static/uploads", 
                 thumbnail_dir: str = "static/thumbnails",
                 compressed_dir: str = "static/compressed"):
        self.upload_dir = upload_dir
        self.thumbnail_dir = thumbnail_dir
        self.compressed_dir = compressed_dir
        
        # Create directories if they don't exist
        for directory in [upload_dir, thumbnail_dir, compressed_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_camera_image(self, session_id: str, image_data: bytes, 
                           camera_settings: Dict[str, Any], user_id: str,
                           filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a camera-captured image through the complete pipeline.
        
        Args:
            session_id: Camera session identifier
            image_data: Raw image data bytes
            camera_settings: Camera settings used for capture
            user_id: User identifier
            filename: Optional original filename
            
        Returns:
            Dict containing processing results
        """
        
        # Create captured image record
        image = CapturedImage(
            session_id=session_id,
            user_id=user_id,
            original_filename=filename or f"camera_capture_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg",
            file_size_bytes=len(image_data),
            camera_settings=CameraSettings.from_dict(camera_settings)
        )
        
        try:
            # Step 1: Validate and parse image
            self._create_processing_step(image.image_id, "image_validation", 1)
            pil_image = self._validate_and_parse_image(image_data, image.image_id)
            
            # Update image metadata
            image.width = pil_image.width
            image.height = pil_image.height
            image.image_format = pil_image.format or "JPEG"
            
            # Extract EXIF data
            image.exif_data = self._extract_exif_data(pil_image)
            
            # Step 2: Quality validation
            self._create_processing_step(image.image_id, "quality_validation", 2)
            quality_result = self._validate_image_quality(pil_image, image.image_id)
            image.set_quality_score(quality_result['overall_score'], quality_result['checks'])
            
            # Step 3: Save original image
            self._create_processing_step(image.image_id, "save_original", 3)
            image.storage_path = self._save_original_image(pil_image, image.image_id, image.image_format)
            
            # Step 4: Generate thumbnail
            self._create_processing_step(image.image_id, "generate_thumbnail", 4)
            image.thumbnail_path = self._generate_thumbnail(pil_image, image.image_id)
            
            # Step 5: Generate compressed version if needed
            if quality_result['needs_compression']:
                self._create_processing_step(image.image_id, "compress_image", 5)
                image.compressed_path = self._compress_image(pil_image, image.image_id)
            
            # Step 6: Prepare for analysis
            self._create_processing_step(image.image_id, "prepare_analysis", 6)
            analysis_ready = self._prepare_for_analysis(pil_image, image.image_id)
            
            # Update processing status
            image.update_processing_status(ProcessingStatus.COMPLETED)
            
            # Save to storage
            storage.save_image(image)
            
            return {
                'image_id': image.image_id,
                'session_id': session_id,
                'processing_status': image.processing_status.value,
                'storage_path': image.storage_path,
                'thumbnail_path': image.thumbnail_path,
                'compressed_path': image.compressed_path,
                'quality_score': image.quality_score,
                'quality_checks': image.quality_checks,
                'analysis_ready': analysis_ready,
                'width': image.width,
                'height': image.height,
                'file_size_bytes': image.file_size_bytes
            }
            
        except Exception as e:
            # Update processing status to failed
            image.update_processing_status(ProcessingStatus.FAILED)
            storage.save_image(image)
            
            # Create failure step
            step = ProcessingStep(
                image_id=image.image_id,
                step_name="processing_failed",
                step_order=999
            )
            step.fail_step(str(e))
            storage.save_processing_step(step)
            
            raise Exception(f"Image processing failed: {str(e)}")
    
    def _validate_and_parse_image(self, image_data: bytes, image_id: str) -> Image.Image:
        """Validate and parse image data"""
        step = self._get_current_step(image_id, "image_validation")
        step.start_step()
        
        try:
            # Try to open image with PIL
            image_stream = io.BytesIO(image_data)
            pil_image = Image.open(image_stream)
            
            # Verify image can be loaded
            pil_image.verify()
            
            # Reopen for actual processing
            image_stream.seek(0)
            pil_image = Image.open(image_stream)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            step.complete_step({
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size': pil_image.size
            })
            
            return pil_image
            
        except Exception as e:
            step.fail_step(f"Image validation failed: {str(e)}")
            raise ValueError(f"ImageFormatError: {str(e)}")
    
    def _validate_image_quality(self, pil_image: Image.Image, image_id: str) -> Dict[str, Any]:
        """Validate image quality for disease analysis"""
        step = self._get_current_step(image_id, "quality_validation")
        step.start_step()
        
        try:
            # Convert PIL to OpenCV format for analysis
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Quality checks
            checks = {}
            
            # 1. Size check
            width, height = pil_image.size
            min_size = 224  # Minimum size for ResNet analysis
            size_score = min(1.0, min(width, height) / min_size) if min(width, height) >= min_size else 0.0
            checks['size'] = QualityCheckResult(
                check_name='size',
                score=size_score,
                status=self._score_to_status(size_score),
                details={'width': width, 'height': height, 'min_required': min_size}
            ).to_dict()
            
            # 2. Brightness check
            brightness_score = self._check_brightness(cv_image)
            checks['brightness'] = QualityCheckResult(
                check_name='brightness',
                score=brightness_score,
                status=self._score_to_status(brightness_score)
            ).to_dict()
            
            # 3. Sharpness check (Laplacian variance)
            sharpness_score = self._check_sharpness(cv_image)
            checks['sharpness'] = QualityCheckResult(
                check_name='sharpness',
                score=sharpness_score,
                status=self._score_to_status(sharpness_score)
            ).to_dict()
            
            # 4. Contrast check
            contrast_score = self._check_contrast(cv_image)
            checks['contrast'] = QualityCheckResult(
                check_name='contrast',
                score=contrast_score,
                status=self._score_to_status(contrast_score)
            ).to_dict()
            
            # 5. Plant detection check (simple green pixel analysis)
            plant_score = self._check_plant_presence(cv_image)
            checks['plant_detection'] = QualityCheckResult(
                check_name='plant_detection',
                score=plant_score,
                status=self._score_to_status(plant_score)
            ).to_dict()
            
            # Calculate overall score (weighted average)
            weights = {
                'size': 0.2,
                'brightness': 0.2,
                'sharpness': 0.25,
                'contrast': 0.15,
                'plant_detection': 0.2
            }
            
            overall_score = sum(checks[check]['score'] * weights[check] for check in weights)
            
            # Determine if compression is needed
            file_size_mb = len(cv_image.tobytes()) / (1024 * 1024)
            needs_compression = file_size_mb > 2.0 or max(width, height) > 2048
            
            result = {
                'overall_score': overall_score,
                'checks': checks,
                'needs_compression': needs_compression,
                'recommendations': self._generate_quality_recommendations(checks)
            }
            
            step.complete_step(result)
            return result
            
        except Exception as e:
            step.fail_step(f"Quality validation failed: {str(e)}")
            raise ValueError(f"ImageQualityError: {str(e)}")
    
    def _check_brightness(self, cv_image: np.ndarray) -> float:
        """Check image brightness"""
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # Optimal brightness is around 0.4-0.7
        if 0.4 <= brightness <= 0.7:
            return 1.0
        elif 0.2 <= brightness < 0.4 or 0.7 < brightness <= 0.8:
            return 0.7
        elif 0.1 <= brightness < 0.2 or 0.8 < brightness <= 0.9:
            return 0.4
        else:
            return 0.1
    
    def _check_sharpness(self, cv_image: np.ndarray) -> float:
        """Check image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize sharpness score (typical good values are above 100)
        if laplacian_var > 500:
            return 1.0
        elif laplacian_var > 200:
            return 0.8
        elif laplacian_var > 100:
            return 0.6
        elif laplacian_var > 50:
            return 0.4
        else:
            return 0.2
    
    def _check_contrast(self, cv_image: np.ndarray) -> float:
        """Check image contrast"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std() / 255.0
        
        # Good contrast typically has std > 0.3
        if contrast > 0.4:
            return 1.0
        elif contrast > 0.3:
            return 0.8
        elif contrast > 0.2:
            return 0.6
        elif contrast > 0.1:
            return 0.4
        else:
            return 0.2
    
    def _check_plant_presence(self, cv_image: np.ndarray) -> float:
        """Check for plant presence using simple green pixel analysis"""
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green pixels
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (cv_image.shape[0] * cv_image.shape[1])
        
        # Score based on green pixel ratio
        if green_ratio > 0.3:
            return 1.0
        elif green_ratio > 0.2:
            return 0.8
        elif green_ratio > 0.1:
            return 0.6
        elif green_ratio > 0.05:
            return 0.4
        else:
            return 0.2
    
    def _score_to_status(self, score: float) -> str:
        """Convert numeric score to status string"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_quality_recommendations(self, checks: Dict[str, Dict]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for check_name, check_data in checks.items():
            if check_data['score'] < 0.6:
                if check_name == 'brightness':
                    recommendations.append("Improve lighting conditions or adjust camera exposure")
                elif check_name == 'sharpness':
                    recommendations.append("Hold camera steady and ensure proper focus")
                elif check_name == 'contrast':
                    recommendations.append("Ensure good lighting contrast between plant and background")
                elif check_name == 'plant_detection':
                    recommendations.append("Position camera closer to plant or ensure plant fills most of frame")
                elif check_name == 'size':
                    recommendations.append("Use higher resolution or move camera closer to subject")
        
        return recommendations
    
    def _save_original_image(self, pil_image: Image.Image, image_id: str, format: str) -> str:
        """Save original image to storage"""
        step = self._get_current_step(image_id, "save_original")
        step.start_step()
        
        try:
            filename = f"{image_id}_original.{format.lower()}"
            file_path = os.path.join(self.upload_dir, filename)
            
            pil_image.save(file_path, format=format, quality=95)
            
            step.complete_step({'file_path': file_path, 'format': format})
            return file_path
            
        except Exception as e:
            step.fail_step(f"Failed to save original image: {str(e)}")
            raise
    
    def _generate_thumbnail(self, pil_image: Image.Image, image_id: str) -> str:
        """Generate thumbnail image"""
        step = self._get_current_step(image_id, "generate_thumbnail")
        step.start_step()
        
        try:
            thumbnail_size = (224, 224)
            thumbnail = pil_image.copy()
            thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            filename = f"{image_id}_thumbnail.jpg"
            file_path = os.path.join(self.thumbnail_dir, filename)
            
            thumbnail.save(file_path, format='JPEG', quality=85)
            
            step.complete_step({'file_path': file_path, 'size': thumbnail_size})
            return file_path
            
        except Exception as e:
            step.fail_step(f"Failed to generate thumbnail: {str(e)}")
            raise
    
    def _compress_image(self, pil_image: Image.Image, image_id: str) -> str:
        """Compress image for analysis"""
        step = self._get_current_step(image_id, "compress_image")
        step.start_step()
        
        try:
            # Resize if too large
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                compressed = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                compressed = pil_image.copy()
            
            filename = f"{image_id}_compressed.jpg"
            file_path = os.path.join(self.compressed_dir, filename)
            
            compressed.save(file_path, format='JPEG', quality=80, optimize=True)
            
            step.complete_step({'file_path': file_path, 'original_size': pil_image.size, 'compressed_size': compressed.size})
            return file_path
            
        except Exception as e:
            step.fail_step(f"Failed to compress image: {str(e)}")
            raise
    
    def _prepare_for_analysis(self, pil_image: Image.Image, image_id: str) -> bool:
        """Prepare image for disease analysis"""
        step = self._get_current_step(image_id, "prepare_analysis")
        step.start_step()
        
        try:
            # Check if image meets minimum requirements for analysis
            min_size = 224
            if min(pil_image.size) < min_size:
                step.complete_step({'analysis_ready': False, 'reason': 'Image too small for analysis'})
                return False
            
            # Additional analysis preparation could go here
            # (e.g., format conversion, metadata preparation)
            
            step.complete_step({'analysis_ready': True})
            return True
            
        except Exception as e:
            step.fail_step(f"Failed to prepare for analysis: {str(e)}")
            return False
    
    def _extract_exif_data(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        exif_data = {}
        
        try:
            if hasattr(pil_image, '_getexif') and pil_image._getexif() is not None:
                exif = pil_image._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)  # Convert to string for JSON serialization
        except Exception:
            # EXIF extraction failed, continue without it
            pass
        
        return exif_data
    
    def _create_processing_step(self, image_id: str, step_name: str, step_order: int) -> ProcessingStep:
        """Create a new processing step"""
        step = ProcessingStep(
            image_id=image_id,
            step_name=step_name,
            step_order=step_order
        )
        storage.save_processing_step(step)
        return step
    
    def _get_current_step(self, image_id: str, step_name: str) -> ProcessingStep:
        """Get the current processing step"""
        steps = storage.get_processing_steps(image_id)
        for step in steps:
            if step.step_name == step_name:
                return step
        
        # If not found, create a new step
        return ProcessingStep(
            image_id=image_id,
            step_name=step_name,
            step_order=len(steps) + 1
        )
    
    def get_image_processing_status(self, image_id: str) -> Dict[str, Any]:
        """Get processing status for an image"""
        image = storage.get_image(image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found")
        
        steps = storage.get_processing_steps(image_id)
        steps.sort(key=lambda x: x.step_order)
        
        return {
            'image_id': image_id,
            'processing_status': image.processing_status.value,
            'quality_score': image.quality_score,
            'quality_checks': image.quality_checks,
            'processing_steps': [step.to_dict() for step in steps],
            'created_at': image.created_at.isoformat(),
            'updated_at': image.updated_at.isoformat()
        }
    
    def validate_image_quality_only(self, image_data: bytes) -> Dict[str, Any]:
        """Validate image quality without full processing"""
        try:
            # Parse image
            image_stream = io.BytesIO(image_data)
            pil_image = Image.open(image_stream)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Run quality validation
            temp_image_id = str(uuid.uuid4())
            quality_result = self._validate_image_quality(pil_image, temp_image_id)
            
            return {
                'is_valid': quality_result['overall_score'] >= 0.5,
                'quality_score': quality_result['overall_score'],
                'quality_checks': quality_result['checks'],
                'recommendations': quality_result['recommendations']
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'quality_checks': {},
                'recommendations': [f"Image validation failed: {str(e)}"]
            }
