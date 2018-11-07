try:  # Just to get rid of warnings on Windows machines
    from picamera import PiCamera
except ImportError:
    PiCamera = None
import time

camera = PiCamera()

camera.start_preview()
time.sleep(10)
camera.stop_preview()

