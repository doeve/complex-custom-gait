# File: src/data/video_processor.py

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from loguru import logger

class VideoProcessor:
    """Handles video processing operations for gait recognition."""
    
    def __init__(self, target_size: Tuple[int, int] = (299, 299)):
        """Initialize video processor.
        
        Args:
            target_size: Target size for frames (height, width)
        """
        self.target_size = target_size

    def extract_frames(self, video_path: str, sampling_rate: int = 1) -> np.ndarray:
        """Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            sampling_rate: Sample every nth frame
            
        Returns:
            Numpy array of frames with shape (num_frames, height, width, channels)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sampling_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                frames.append(frame)
            
            frame_count += 1
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
            
        return np.array(frames)

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frames for the neural network.
        
        Args:
            frames: Numpy array of frames (num_frames, height, width, channels)
            
        Returns:
            Preprocessed frames
        """
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Center around zero
        frames = frames - 0.5
        
        # Scale to [-1, 1]
        frames = frames * 2.0
        
        return frames

    def center_crop_frames(self, frames: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Crop frames to center on the person.
        
        Args:
            frames: Input frames
            crop_size: Desired crop size (height, width)
            
        Returns:
            Cropped frames
        """
        if len(frames.shape) != 4:
            raise ValueError("Expected 4D array of frames")
            
        height, width = frames.shape[1:3]
        
        start_y = (height - crop_size[0]) // 2
        start_x = (width - crop_size[1]) // 2
        
        return frames[:, start_y:start_y + crop_size[0], 
                     start_x:start_x + crop_size[1], :]

    @staticmethod
    def save_debug_frame(frame: np.ndarray, path: str):
        """Save a frame for debugging purposes.
        
        Args:
            frame: Frame to save
            path: Save path
        """
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame_bgr)