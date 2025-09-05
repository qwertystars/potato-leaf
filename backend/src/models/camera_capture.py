"""
Camera Capture Data Models
Defines the data structures for camera upload functionality.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid
import json


class SessionStatus(str, Enum):
    """Camera session status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class ProcessingStatus(str, Enum):
    """Image processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Processing step status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DeviceInfo:
    """Device information for camera sessions"""
    device_type: str  # mobile, tablet, desktop
    browser: str  # Browser name and version
    user_agent: str  # Full user agent string
    screen_resolution: str  # Screen resolution (WxH)
    pixel_ratio: float = 1.0  # Device pixel ratio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'device_type': self.device_type,
            'browser': self.browser,
            'user_agent': self.user_agent,
            'screen_resolution': self.screen_resolution,
            'pixel_ratio': self.pixel_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceInfo':
        """Create from dictionary"""
        return cls(
            device_type=data['device_type'],
            browser=data['browser'],
            user_agent=data['user_agent'],
            screen_resolution=data['screen_resolution'],
            pixel_ratio=data.get('pixel_ratio', 1.0)
        )


@dataclass
class CameraCapabilities:
    """Camera capabilities information"""
    available_cameras: List[Dict[str, Any]] = field(default_factory=list)
    max_resolution: str = ""  # Maximum camera resolution
    supports_autofocus: bool = False
    supports_flash: bool = False
    zoom_range: Optional[str] = None  # Min-max zoom levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'available_cameras': self.available_cameras,
            'max_resolution': self.max_resolution,
            'supports_autofocus': self.supports_autofocus,
            'supports_flash': self.supports_flash,
            'zoom_range': self.zoom_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraCapabilities':
        """Create from dictionary"""
        return cls(
            available_cameras=data.get('available_cameras', []),
            max_resolution=data.get('max_resolution', ''),
            supports_autofocus=data.get('supports_autofocus', False),
            supports_flash=data.get('supports_flash', False),
            zoom_range=data.get('zoom_range')
        )


@dataclass
class CameraSession:
    """Camera session data model"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    device_info: Optional[DeviceInfo] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    total_captures: int = 0
    successful_analyses: int = 0
    session_status: SessionStatus = SessionStatus.ACTIVE
    camera_capabilities: Optional[CameraCapabilities] = None
    selected_camera_id: Optional[str] = None
    location_data: Optional[Dict[str, Any]] = None
    environmental_conditions: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'device_info': self.device_info.to_dict() if self.device_info else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'total_captures': self.total_captures,
            'successful_analyses': self.successful_analyses,
            'session_status': self.session_status.value,
            'camera_capabilities': self.camera_capabilities.to_dict() if self.camera_capabilities else None,
            'selected_camera_id': self.selected_camera_id,
            'location_data': self.location_data,
            'environmental_conditions': self.environmental_conditions,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraSession':
        """Create from dictionary"""
        return cls(
            session_id=data.get('session_id', str(uuid.uuid4())),
            user_id=data.get('user_id', ''),
            device_info=DeviceInfo.from_dict(data['device_info']) if data.get('device_info') else None,
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else datetime.utcnow(),
            ended_at=datetime.fromisoformat(data['ended_at']) if data.get('ended_at') else None,
            total_captures=data.get('total_captures', 0),
            successful_analyses=data.get('successful_analyses', 0),
            session_status=SessionStatus(data.get('session_status', 'active')),
            camera_capabilities=CameraCapabilities.from_dict(data['camera_capabilities']) if data.get('camera_capabilities') else None,
            selected_camera_id=data.get('selected_camera_id'),
            location_data=data.get('location_data'),
            environmental_conditions=data.get('environmental_conditions'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.utcnow()
        )
    
    def end_session(self):
        """Mark session as completed"""
        self.session_status = SessionStatus.COMPLETED
        self.ended_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def increment_capture_count(self):
        """Increment total capture count"""
        self.total_captures += 1
        self.updated_at = datetime.utcnow()
    
    def increment_analysis_count(self):
        """Increment successful analysis count"""
        self.successful_analyses += 1
        self.updated_at = datetime.utcnow()


@dataclass
class CameraSettings:
    """Camera settings for image capture"""
    resolution: str = "1920x1080"
    flash: bool = False
    focus_mode: str = "auto"  # auto, manual, continuous
    zoom_level: float = 1.0
    iso: Optional[int] = None
    white_balance: str = "auto"  # auto, daylight, fluorescent, etc.
    exposure_compensation: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'resolution': self.resolution,
            'flash': self.flash,
            'focus_mode': self.focus_mode,
            'zoom_level': self.zoom_level,
            'iso': self.iso,
            'white_balance': self.white_balance,
            'exposure_compensation': self.exposure_compensation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraSettings':
        """Create from dictionary"""
        return cls(
            resolution=data.get('resolution', '1920x1080'),
            flash=data.get('flash', False),
            focus_mode=data.get('focus_mode', 'auto'),
            zoom_level=data.get('zoom_level', 1.0),
            iso=data.get('iso'),
            white_balance=data.get('white_balance', 'auto'),
            exposure_compensation=data.get('exposure_compensation', 0.0)
        )


@dataclass
class CapturedImage:
    """Captured image data model"""
    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    user_id: str = ""
    original_filename: Optional[str] = None
    file_size_bytes: int = 0
    image_format: str = "JPEG"  # JPEG, PNG, WebP
    width: int = 0
    height: int = 0
    storage_path: str = ""
    thumbnail_path: Optional[str] = None
    compressed_path: Optional[str] = None
    camera_settings: Optional[CameraSettings] = None
    exif_data: Optional[Dict[str, Any]] = None
    capture_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    quality_score: Optional[float] = None
    quality_checks: Optional[Dict[str, Any]] = None
    analysis_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'image_id': self.image_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'original_filename': self.original_filename,
            'file_size_bytes': self.file_size_bytes,
            'image_format': self.image_format,
            'width': self.width,
            'height': self.height,
            'storage_path': self.storage_path,
            'thumbnail_path': self.thumbnail_path,
            'compressed_path': self.compressed_path,
            'camera_settings': self.camera_settings.to_dict() if self.camera_settings else None,
            'exif_data': self.exif_data,
            'capture_timestamp': self.capture_timestamp.isoformat() if self.capture_timestamp else None,
            'processing_status': self.processing_status.value,
            'quality_score': self.quality_score,
            'quality_checks': self.quality_checks,
            'analysis_id': self.analysis_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapturedImage':
        """Create from dictionary"""
        return cls(
            image_id=data.get('image_id', str(uuid.uuid4())),
            session_id=data.get('session_id', ''),
            user_id=data.get('user_id', ''),
            original_filename=data.get('original_filename'),
            file_size_bytes=data.get('file_size_bytes', 0),
            image_format=data.get('image_format', 'JPEG'),
            width=data.get('width', 0),
            height=data.get('height', 0),
            storage_path=data.get('storage_path', ''),
            thumbnail_path=data.get('thumbnail_path'),
            compressed_path=data.get('compressed_path'),
            camera_settings=CameraSettings.from_dict(data['camera_settings']) if data.get('camera_settings') else None,
            exif_data=data.get('exif_data'),
            capture_timestamp=datetime.fromisoformat(data['capture_timestamp']) if data.get('capture_timestamp') else datetime.utcnow(),
            processing_status=ProcessingStatus(data.get('processing_status', 'pending')),
            quality_score=data.get('quality_score'),
            quality_checks=data.get('quality_checks'),
            analysis_id=data.get('analysis_id'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.utcnow()
        )
    
    def update_processing_status(self, status: ProcessingStatus):
        """Update processing status"""
        self.processing_status = status
        self.updated_at = datetime.utcnow()
    
    def set_quality_score(self, score: float, checks: Dict[str, Any]):
        """Set quality score and checks"""
        self.quality_score = score
        self.quality_checks = checks
        self.updated_at = datetime.utcnow()
    
    def set_analysis_id(self, analysis_id: str):
        """Set analysis ID when submitted for analysis"""
        self.analysis_id = analysis_id
        self.updated_at = datetime.utcnow()


@dataclass
class ProcessingStep:
    """Image processing step data model"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str = ""
    step_name: str = ""
    step_order: int = 0
    algorithm_used: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    input_image_path: Optional[str] = None
    output_image_path: Optional[str] = None
    step_status: StepStatus = StepStatus.PENDING
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'step_id': self.step_id,
            'image_id': self.image_id,
            'step_name': self.step_name,
            'step_order': self.step_order,
            'algorithm_used': self.algorithm_used,
            'parameters': self.parameters,
            'execution_time_ms': self.execution_time_ms,
            'input_image_path': self.input_image_path,
            'output_image_path': self.output_image_path,
            'step_status': self.step_status.value,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def start_step(self):
        """Mark step as running"""
        self.step_status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_step(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark step as completed"""
        self.step_status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if result_data:
            self.result_data = result_data
        if self.started_at:
            self.execution_time_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def fail_step(self, error_message: str):
        """Mark step as failed"""
        self.step_status = StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message


@dataclass
class QualityCheckResult:
    """Quality check result data model"""
    check_name: str
    score: float  # 0.0 to 1.0
    status: str  # poor, fair, good, excellent
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'check_name': self.check_name,
            'score': self.score,
            'status': self.status,
            'details': self.details,
            'recommendations': self.recommendations or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityCheckResult':
        """Create from dictionary"""
        return cls(
            check_name=data['check_name'],
            score=data['score'],
            status=data['status'],
            details=data.get('details'),
            recommendations=data.get('recommendations', [])
        )


# Simple in-memory storage for development/testing
class InMemoryStorage:
    """Simple in-memory storage for camera upload data"""
    
    def __init__(self):
        self.sessions: Dict[str, CameraSession] = {}
        self.images: Dict[str, CapturedImage] = {}
        self.processing_steps: Dict[str, List[ProcessingStep]] = {}
    
    def save_session(self, session: CameraSession) -> None:
        """Save camera session"""
        self.sessions[session.session_id] = session
    
    def get_session(self, session_id: str) -> Optional[CameraSession]:
        """Get camera session by ID"""
        return self.sessions.get(session_id)
    
    def save_image(self, image: CapturedImage) -> None:
        """Save captured image"""
        self.images[image.image_id] = image
    
    def get_image(self, image_id: str) -> Optional[CapturedImage]:
        """Get captured image by ID"""
        return self.images.get(image_id)
    
    def save_processing_step(self, step: ProcessingStep) -> None:
        """Save processing step"""
        if step.image_id not in self.processing_steps:
            self.processing_steps[step.image_id] = []
        self.processing_steps[step.image_id].append(step)
    
    def get_processing_steps(self, image_id: str) -> List[ProcessingStep]:
        """Get processing steps for image"""
        return self.processing_steps.get(image_id, [])
    
    def clear(self) -> None:
        """Clear all stored data"""
        self.sessions.clear()
        self.images.clear()
        self.processing_steps.clear()


# Global storage instance for development
storage = InMemoryStorage()
