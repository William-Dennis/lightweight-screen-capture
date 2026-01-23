import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from lightweight_screen_capture.screen_capture import ScreenCapturer

@pytest.fixture
def mock_mss():
    """Mocks the mss.mss() constructor and its instance."""
    with patch('mss.mss') as mocked_class:
        mock_instance = MagicMock()
        # Mock monitors list: [All, Mon1]
        mock_instance.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
        
        # Setup mock grab to return an object that converts to a numpy array
        mock_shot = MagicMock()
        mock_shot.__array__ = MagicMock(return_value=np.zeros((100, 100, 4), dtype=np.uint8))
        mock_instance.grab.return_value = mock_shot
        
        mocked_class.return_value = mock_instance
        yield mock_instance

def test_init_does_not_crash():
    """Verify that importing and initializing doesn't hit hardware immediately."""
    with patch('mss.mss'):
        capturer = ScreenCapturer()
        assert capturer is not None

def test_capture_screen_logic(mock_mss):
    capturer = ScreenCapturer()
    result = capturer.capture_screen(monitor=1)
    
    assert result.shape == (100, 100, 4)
    mock_mss.grab.assert_called_once_with(mock_mss.monitors[1])

def test_capture_window_math(mock_mss):
    with patch('win32gui.FindWindow', return_value=123), \
         patch('win32gui.GetWindowRect', return_value=(10, 10, 60, 60)): # 50x50 size
        
        capturer = ScreenCapturer()
        capturer.capture_window("Any Window")
        
        expected_region = {"left": 10, "top": 10, "width": 50, "height": 50}
        mock_mss.grab.assert_called_with(expected_region)

def test_capture_window_not_found():
    with patch('win32gui.FindWindow', return_value=0):
        capturer = ScreenCapturer()
        with pytest.raises(RuntimeError, match="not found"):
            capturer.capture_window("NonExistent")