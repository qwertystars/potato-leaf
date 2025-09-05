"""
Camera Upload API Routes for Flask Application
Handles camera session management and image capture endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
import uuid
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..models.camera_capture import (
    CameraSession, DeviceInfo, CameraCapabilities, 
    CapturedImage, SessionStatus, storage
)
from ..services.camera_image_service import CameraImageService


# Create Blueprint for camera routes
camera_bp = Blueprint('camera', __name__, url_prefix='/api/camera')

# Initialize camera image service
camera_service = CameraImageService()


@camera_bp.route('/session', methods=['POST'])
def initialize_camera_session():
    """Initialize a new camera session"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        # Validate required fields
        if 'device_info' not in data or 'camera_capabilities' not in data:
            return jsonify({'error': 'device_info and camera_capabilities are required'}), 400
        
        # Create device info
        device_info = DeviceInfo.from_dict(data['device_info'])
        
        # Create camera capabilities
        camera_capabilities = CameraCapabilities.from_dict(data['camera_capabilities'])
        
        # Generate user ID (in real app, this would come from authentication)
        user_id = data.get('user_id', f"user_{uuid.uuid4().hex[:8]}")
        
        # Create camera session
        session = CameraSession(
            user_id=user_id,
            device_info=device_info,
            camera_capabilities=camera_capabilities,
            location_data=data.get('location_data'),
            environmental_conditions=data.get('environmental_conditions')
        )
        
        # Save session
        storage.save_session(session)
        
        current_app.logger.info(f"Camera session initialized: {session.session_id} for user {user_id}")
        
        return jsonify(session.to_dict()), 201
        
    except Exception as e:
        current_app.logger.error(f"Failed to initialize camera session: {str(e)}")
        return jsonify({'error': f'Failed to initialize session: {str(e)}'}), 500


@camera_bp.route('/session/<session_id>', methods=['GET'])
def get_camera_session(session_id: str):
    """Get camera session details"""
    try:
        session = storage.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(session.to_dict()), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to get camera session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to get session: {str(e)}'}), 500


@camera_bp.route('/session/<session_id>', methods=['PATCH'])
def update_camera_session(session_id: str):
    """Update camera session (e.g., end session)"""
    try:
        session = storage.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        # Update session status if provided
        if 'session_status' in data:
            if data['session_status'] == 'completed':
                session.end_session()
            else:
                session.session_status = SessionStatus(data['session_status'])
                session.updated_at = datetime.utcnow()
        
        # Update other fields if provided
        if 'selected_camera_id' in data:
            session.selected_camera_id = data['selected_camera_id']
            session.updated_at = datetime.utcnow()
        
        # Save updated session
        storage.save_session(session)
        
        current_app.logger.info(f"Camera session updated: {session_id}")
        
        return jsonify(session.to_dict()), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to update camera session {session_id}: {str(e)}")
        return jsonify({'error': f'Failed to update session: {str(e)}'}), 500


@camera_bp.route('/capture', methods=['POST'])
def capture_image():
    """Process a captured image from camera"""
    try:
        # Get session ID
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Verify session exists
        session = storage.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get uploaded image file
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get camera settings
        camera_settings_str = request.form.get('camera_settings', '{}')
        try:
            camera_settings = json.loads(camera_settings_str)
        except json.JSONDecodeError:
            camera_settings = {}
        
        # Read image data
        image_data = image_file.read()
        
        # Process the image
        result = camera_service.process_camera_image(
            session_id=session_id,
            image_data=image_data,
            camera_settings=camera_settings,
            user_id=session.user_id,
            filename=image_file.filename
        )
        
        # Update session capture count
        session.increment_capture_count()
        if result.get('analysis_ready', False):
            session.increment_analysis_count()
        storage.save_session(session)
        
        current_app.logger.info(f"Image captured and processed: {result['image_id']} in session {session_id}")
        
        return jsonify(result), 201
        
    except Exception as e:
        current_app.logger.error(f"Failed to process captured image: {str(e)}")
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500


@camera_bp.route('/image/<image_id>/status', methods=['GET'])
def get_image_processing_status(image_id: str):
    """Get image processing status"""
    try:
        status = camera_service.get_image_processing_status(image_id)
        return jsonify(status), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Failed to get image processing status {image_id}: {str(e)}")
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500


@camera_bp.route('/validate', methods=['POST'])
def validate_image_quality():
    """Validate image quality without full processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Validate quality
        result = camera_service.validate_image_quality_only(image_data)
        
        current_app.logger.info(f"Image quality validated: score={result.get('quality_score', 0)}")
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to validate image quality: {str(e)}")
        return jsonify({'error': f'Failed to validate image: {str(e)}'}), 500


@camera_bp.route('/settings/optimal', methods=['GET'])
def get_optimal_camera_settings():
    """Get optimal camera settings for disease analysis"""
    try:
        device_type = request.args.get('device_type', 'mobile')
        lighting_conditions = request.args.get('lighting_conditions', 'normal')
        
        # Generate optimal settings based on device and conditions
        settings = _generate_optimal_settings(device_type, lighting_conditions)
        
        return jsonify(settings), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to get optimal camera settings: {str(e)}")
        return jsonify({'error': f'Failed to get settings: {str(e)}'}), 500


@camera_bp.route('/permissions', methods=['GET'])
def check_camera_permissions():
    """Check camera permissions and availability (client-side check)"""
    try:
        # This is primarily a client-side check, but we can provide guidance
        user_agent = request.headers.get('User-Agent', '')
        
        # Basic browser support detection
        browser_support = _detect_browser_support(user_agent)
        
        response = {
            'camera_available': True,  # Assume true, real check is client-side
            'permission_granted': None,  # Cannot check server-side
            'available_cameras': [],  # Cannot enumerate server-side
            'browser_support': browser_support,
            'recommendations': _get_camera_recommendations(user_agent)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to check camera permissions: {str(e)}")
        return jsonify({'error': f'Failed to check permissions: {str(e)}'}), 500


def _generate_optimal_settings(device_type: str, lighting_conditions: str) -> Dict[str, Any]:
    """Generate optimal camera settings based on device and conditions"""
    
    base_settings = {
        'resolution': '1920x1080',
        'focus_mode': 'auto',
        'flash_enabled': False,
        'iso_settings': 'auto',
        'white_balance': 'auto'
    }
    
    # Adjust based on device type
    if device_type == 'mobile':
        base_settings['resolution'] = '1280x720'  # Lower resolution for mobile
    elif device_type == 'tablet':
        base_settings['resolution'] = '1920x1080'
    elif device_type == 'desktop':
        base_settings['resolution'] = '1920x1080'
    
    # Adjust based on lighting conditions
    if lighting_conditions == 'dim':
        base_settings['flash_enabled'] = True
        base_settings['iso_settings'] = '400-800'
    elif lighting_conditions == 'bright':
        base_settings['iso_settings'] = '100-200'
    elif lighting_conditions == 'artificial':
        base_settings['white_balance'] = 'fluorescent'
    
    return base_settings


def _detect_browser_support(user_agent: str) -> Dict[str, Any]:
    """Detect browser support for camera APIs"""
    
    support = {
        'getUserMedia': True,  # Assume modern browser
        'MediaDevices': True,
        'WebRTC': True,
        'FileAPI': True
    }
    
    # Basic detection (in real app, would be more sophisticated)
    if 'Chrome' in user_agent:
        support['browser'] = 'Chrome'
        support['version'] = 'Modern'
    elif 'Firefox' in user_agent:
        support['browser'] = 'Firefox'
        support['version'] = 'Modern'
    elif 'Safari' in user_agent:
        support['browser'] = 'Safari'
        support['version'] = 'Modern'
    elif 'Edge' in user_agent:
        support['browser'] = 'Edge'
        support['version'] = 'Modern'
    else:
        support['browser'] = 'Unknown'
        support['version'] = 'Unknown'
    
    return support


def _get_camera_recommendations(user_agent: str) -> List[str]:
    """Get camera usage recommendations"""
    
    recommendations = [
        "Ensure good lighting for best image quality",
        "Hold camera steady to avoid blur",
        "Position plant to fill most of the frame",
        "Use rear camera for better quality (mobile devices)"
    ]
    
    # Add device-specific recommendations
    if 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent:
        recommendations.extend([
            "Clean camera lens before taking photos",
            "Use landscape orientation for better framing",
            "Tap to focus on the plant before capturing"
        ])
    
    return recommendations


# Error handlers for the blueprint
@camera_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400


@camera_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@camera_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
