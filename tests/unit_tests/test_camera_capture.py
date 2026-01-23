import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from lightweight_screen_capture.camera_capture import CameraManager

@pytest.fixture
def mock_cv2_cap():
    """Mocks the cv2.VideoCapture instance."""
    with patch('cv2.VideoCapture') as mocked_vc:
        mock_instance = MagicMock()
        mocked_vc.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def fake_frame():
    """Generates a dummy BGR image (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

# --- NEW TESTS ---

def test_camera_initialization_success(mock_cv2_cap):
    """Verify camera opens correctly with the right index."""
    mock_cv2_cap.isOpened.return_value = True
    
    cam = CameraManager(index=1)
    
    assert cam.cap is not None
    mock_cv2_cap.isOpened.assert_called_once()

def test_camera_initialization_failure(mock_cv2_cap):
    """Verify error handling when camera cannot be opened."""
    mock_cv2_cap.isOpened.return_value = False
    
    with pytest.raises(RuntimeError, match="Cannot open camera at index 0"):
        CameraManager(index=0)

def test_capture_frame_dimensions(mock_cv2_cap, fake_frame):
    """Verify the captured frame retains its shape and type."""
    mock_cv2_cap.isOpened.return_value = True
    mock_cv2_cap.read.return_value = (True, fake_frame)
    
    cam = CameraManager()
    frame = cam.capture_frame(flip=False)
    
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8

def test_horizontal_flip_logic(mock_cv2_cap):
    """Verify that cv2.flip is actually called when requested."""
    mock_cv2_cap.isOpened.return_value = True
    # Create a frame with a distinct left side to test flipping
    left_heavy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    left_heavy_frame[:, :50] = 255 
    
    mock_cv2_cap.read.return_value = (True, left_heavy_frame)
    
    with patch('cv2.flip') as mock_flip:
        # We simulate the flip by returning a dummy
        mock_flip.return_value = "flipped_image"
        
        cam = CameraManager()
        result = cam.capture_frame(flip=True)
        
        assert result == "flipped_image"
        mock_flip.assert_called_once_with(left_heavy_frame, 1)

def test_context_manager_cleanup(mock_cv2_cap):
    """Verify the camera is released automatically using the 'with' statement."""
    mock_cv2_cap.isOpened.return_value = True
    
    with CameraManager() as cam:
        pass  # Do nothing
    
    # Check if release was called upon exiting the 'with' block
    mock_cv2_cap.release.assert_called_once()