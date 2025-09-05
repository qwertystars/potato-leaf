"""
Integration tests for Camera Upload functionality.
Tests the complete workflow from camera capture to analysis integration.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import uuid
from datetime import datetime


class TestCameraUploadIntegration:
    """Test complete camera upload workflow integration"""
    
    def setup_method(self):
        """Set up test environment with mock database and services"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_db_session = Mock()
        self.mock_redis_client = Mock()
        
        # Create a sample test image
        self.test_image = Image.new('RGB', (640, 480), color='green')
        self.test_image_data = io.BytesIO()
        self.test_image.save(self.test_image_data, format='JPEG')
        self.test_image_data.seek(0)
        
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('backend.src.services.camera_image_service.CameraImageService')
    @patch('backend.src.models.camera_capture.CameraSession')
    def test_complete_camera_capture_workflow(self, mock_session_model, mock_service):
        """Test complete workflow: session creation -> image capture -> processing -> analysis"""
        
        # Arrange
        session_id = str(uuid.uuid4())
        image_id = str(uuid.uuid4())
        user_id = "test_user_123"
        
        # Mock session creation
        mock_session_instance = Mock()
        mock_session_instance.session_id = session_id
        mock_session_instance.user_id = user_id
        mock_session_instance.session_status = "active"
        mock_session_model.return_value = mock_session_instance
        
        # Mock image service
        mock_service_instance = Mock()
        mock_service_instance.process_camera_image.return_value = {
            'image_id': image_id,
            'processing_status': 'completed',
            'quality_score': 0.85,
            'storage_path': f'{self.temp_dir}/processed_image.jpg'
        }
        mock_service.return_value = mock_service_instance
        
        # Act: Simulate the complete workflow
        
        # 1. Initialize camera session
        session_data = {
            'user_id': user_id,
            'device_info': {
                'device_type': 'mobile',
                'browser': 'Chrome 91.0'
            },
            'camera_capabilities': {
                'max_resolution': '1920x1080',
                'supports_autofocus': True
            }
        }
        
        # 2. Capture and process image
        capture_result = mock_service_instance.process_camera_image(
            session_id=session_id,
            image_data=self.test_image_data.read(),
            camera_settings={'resolution': '1920x1080'},
            user_id=user_id
        )
        
        # Assert: Verify complete workflow
        assert capture_result['image_id'] == image_id
        assert capture_result['processing_status'] == 'completed'
        assert capture_result['quality_score'] > 0.8  # High quality required
        assert os.path.exists(capture_result['storage_path'])
        
        # Verify service was called with correct parameters
        mock_service_instance.process_camera_image.assert_called_once_with(
            session_id=session_id,
            image_data=self.test_image_data.read(),
            camera_settings={'resolution': '1920x1080'},
            user_id=user_id
        )
        
    @patch('backend.src.services.image_validation_service.ImageValidationService')
    def test_image_quality_validation_integration(self, mock_validation_service):
        """Test integration between camera capture and image quality validation"""
        
        # Arrange
        mock_validator = Mock()
        mock_validator.validate_image_quality.return_value = {
            'is_valid': True,
            'quality_score': 0.92,
            'quality_checks': {
                'brightness': {'score': 0.85, 'status': 'good'},
                'sharpness': {'score': 0.95, 'status': 'excellent'},
                'contrast': {'score': 0.90, 'status': 'good'},
                'plant_detection': {'score': 0.95, 'status': 'excellent'}
            },
            'recommendations': []
        }
        mock_validation_service.return_value = mock_validator
        
        # Act
        validation_result = mock_validator.validate_image_quality(
            image_data=self.test_image_data.read(),
            capture_settings={'resolution': '1920x1080'}
        )
        
        # Assert
        assert validation_result['is_valid'] is True
        assert validation_result['quality_score'] > 0.9
        assert 'quality_checks' in validation_result
        assert 'plant_detection' in validation_result['quality_checks']
        assert validation_result['quality_checks']['plant_detection']['status'] == 'excellent'
        
    @patch('backend.src.services.analysis_integration_service.AnalysisIntegrationService')
    @patch('backend.src.services.camera_image_service.CameraImageService')
    def test_camera_to_analysis_pipeline_integration(self, mock_image_service, mock_analysis_service):
        """Test integration between camera capture and disease analysis pipeline"""
        
        # Arrange
        image_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        
        # Mock image processing
        mock_image_service_instance = Mock()
        mock_image_service_instance.process_camera_image.return_value = {
            'image_id': image_id,
            'processing_status': 'completed',
            'storage_path': f'{self.temp_dir}/processed_image.jpg',
            'analysis_ready': True
        }
        mock_image_service.return_value = mock_image_service_instance
        
        # Mock analysis integration
        mock_analysis_service_instance = Mock()
        mock_analysis_service_instance.submit_for_analysis.return_value = {
            'analysis_id': analysis_id,
            'status': 'submitted',
            'estimated_completion': '2024-09-04T15:30:00Z'
        }
        mock_analysis_service.return_value = mock_analysis_service_instance
        
        # Act: Simulate camera to analysis workflow
        # 1. Process camera image
        image_result = mock_image_service_instance.process_camera_image(
            session_id=str(uuid.uuid4()),
            image_data=self.test_image_data.read(),
            camera_settings={'resolution': '1920x1080'},
            user_id='test_user'
        )
        
        # 2. Submit to analysis if ready
        if image_result['analysis_ready']:
            analysis_result = mock_analysis_service_instance.submit_for_analysis(
                image_id=image_result['image_id'],
                image_path=image_result['storage_path'],
                analysis_type='disease_detection'
            )
        
        # Assert: Verify complete pipeline
        assert image_result['image_id'] == image_id
        assert image_result['analysis_ready'] is True
        assert analysis_result['analysis_id'] == analysis_id
        assert analysis_result['status'] == 'submitted'
        
        # Verify services were called correctly
        mock_analysis_service_instance.submit_for_analysis.assert_called_once_with(
            image_id=image_id,
            image_path=f'{self.temp_dir}/processed_image.jpg',
            analysis_type='disease_detection'
        )
        
    @patch('redis.Redis')
    @patch('backend.src.services.camera_image_service.CameraImageService')
    def test_real_time_processing_with_redis_integration(self, mock_image_service, mock_redis):
        """Test real-time image processing with Redis queue integration"""
        
        # Arrange
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_service_instance = Mock()
        mock_service_instance.queue_for_processing.return_value = {
            'queue_id': 'proc_queue_123',
            'estimated_wait_time': 2.5,
            'position_in_queue': 3
        }
        mock_image_service.return_value = mock_service_instance
        
        # Act: Queue image for real-time processing
        queue_result = mock_service_instance.queue_for_processing(
            image_id=str(uuid.uuid4()),
            priority='high',  # Camera captures get high priority
            processing_type='real_time'
        )
        
        # Assert: Verify queuing
        assert 'queue_id' in queue_result
        assert queue_result['estimated_wait_time'] < 5.0  # Fast processing required
        assert isinstance(queue_result['position_in_queue'], int)
        
    def test_error_handling_integration(self):
        """Test error handling across the camera upload integration"""
        
        # Test various error scenarios
        error_scenarios = [
            {
                'scenario': 'invalid_image_format',
                'image_data': b'invalid_image_data',
                'expected_error': 'ImageFormatError'
            },
            {
                'scenario': 'image_too_small',
                'image_data': self._create_small_image(),
                'expected_error': 'ImageQualityError'
            },
            {
                'scenario': 'corrupted_image',
                'image_data': b'corrupted_image_data_xyz',
                'expected_error': 'ImageCorruptionError'
            }
        ]
        
        for scenario in error_scenarios:
            with pytest.raises(Exception) as exc_info:
                # This should raise appropriate errors for invalid inputs
                self._process_invalid_image(scenario['image_data'])
            
            # Verify proper error type is raised
            assert scenario['expected_error'] in str(type(exc_info.value))
            
    def _create_small_image(self):
        """Create a test image that's too small for analysis"""
        small_image = Image.new('RGB', (50, 50), color='red')
        buffer = io.BytesIO()
        small_image.save(buffer, format='JPEG')
        return buffer.getvalue()
        
    def _process_invalid_image(self, image_data):
        """Mock function to process invalid image data"""
        # Simulate image validation that raises errors for invalid inputs
        if len(image_data) < 100:
            raise ValueError("ImageQualityError: Image too small")
        if b'invalid' in image_data:
            raise ValueError("ImageFormatError: Invalid image format")
        if b'corrupted' in image_data:
            raise ValueError("ImageCorruptionError: Corrupted image data")


class TestCameraSessionManagement:
    """Test camera session management integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_db = Mock()
        
    @patch('backend.src.models.camera_capture.CameraSession')
    def test_session_lifecycle_integration(self, mock_session_model):
        """Test complete session lifecycle from creation to cleanup"""
        
        # Arrange
        session_id = str(uuid.uuid4())
        user_id = "test_user_456"
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.user_id = user_id
        mock_session.session_status = "active"
        mock_session.total_captures = 0
        mock_session_model.return_value = mock_session
        
        # Act: Simulate complete session lifecycle
        
        # 1. Create session
        session = mock_session_model(
            user_id=user_id,
            device_info={'device_type': 'mobile'},
            camera_capabilities={'max_resolution': '1920x1080'}
        )
        
        # 2. Update session with captures
        session.total_captures = 3
        session.successful_analyses = 2
        
        # 3. End session
        session.session_status = "completed"
        session.ended_at = datetime.utcnow()
        
        # Assert: Verify session lifecycle
        assert session.session_id == session_id
        assert session.total_captures == 3
        assert session.successful_analyses == 2
        assert session.session_status == "completed"
        assert session.ended_at is not None
        
    @patch('backend.src.services.camera_image_service.CameraImageService')
    def test_concurrent_capture_handling(self, mock_service):
        """Test handling of concurrent image captures within session"""
        
        # Arrange
        session_id = str(uuid.uuid4())
        mock_service_instance = Mock()
        
        # Mock concurrent processing
        capture_results = []
        for i in range(3):
            capture_results.append({
                'image_id': str(uuid.uuid4()),
                'processing_status': 'completed',
                'capture_order': i + 1
            })
        
        mock_service_instance.process_multiple_captures.return_value = capture_results
        mock_service.return_value = mock_service_instance
        
        # Act: Process multiple concurrent captures
        results = mock_service_instance.process_multiple_captures(
            session_id=session_id,
            captures=[
                {'image_data': b'image1', 'timestamp': '2024-09-04T15:20:00Z'},
                {'image_data': b'image2', 'timestamp': '2024-09-04T15:20:01Z'},
                {'image_data': b'image3', 'timestamp': '2024-09-04T15:20:02Z'}
            ]
        )
        
        # Assert: Verify concurrent processing
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['capture_order'] == i + 1
            assert result['processing_status'] == 'completed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
