# File: src/data/data_loader.py

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

@dataclass
class User:
    """User data structure."""
    id: int
    name: str
    training_videos: List[str]
    gait_signature: Optional[np.ndarray] = None
    last_trained: Optional[str] = None

class DataLoader:
    """Handles data loading and saving operations."""
    
    def __init__(self, data_dir: str):
        """Initialize the data loader.
        
        Args:
            data_dir: Base directory for all data
        """
        self.data_dir = Path(data_dir)
        self.users_file = self.data_dir / 'users' / 'user_database.json'
        self.training_dir = self.data_dir / 'training_videos'
        self.test_dir = self.data_dir / 'test_videos'
        
        # Create directories if they don't exist
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def load_user_database(self) -> Dict[int, User]:
        """Load the user database from disk.
        
        Returns:
            Dictionary mapping user IDs to User objects
        """
        if not self.users_file.exists():
            return {}
            
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
            
            users = {}
            for user_id, user_data in data['users'].items():
                user_id = int(user_id)
                users[user_id] = User(
                    id=user_id,
                    name=user_data['name'],
                    training_videos=user_data['training_videos'],
                    last_trained=user_data.get('last_trained')
                )
                
                # Load gait signature if it exists
                sig_path = self.data_dir / 'users' / f'signature_{user_id}.npy'
                if sig_path.exists():
                    users[user_id].gait_signature = np.load(sig_path)
                    
            return users
            
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
            return {}

    def save_user_database(self, users: Dict[int, User]):
        """Save the user database to disk.
        
        Args:
            users: Dictionary mapping user IDs to User objects
        """
        data = {
            'users': {
                str(user.id): {
                    'name': user.name,
                    'training_videos': user.training_videos,
                    'last_trained': user.last_trained
                }
                for user in users.values()
            }
        }
        
        try:
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Save gait signatures separately
            for user in users.values():
                if user.gait_signature is not None:
                    sig_path = self.data_dir / 'users' / f'signature_{user.id}.npy'
                    np.save(sig_path, user.gait_signature)
                    
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
            raise

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise

    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save a configuration dictionary to YAML.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the config file
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving config file {config_path}: {e}")
            raise

    def get_video_path(self, user_id: int, video_name: str, is_training: bool = True) -> Path:
        """Get the full path for a video file.
        
        Args:
            user_id: ID of the user
            video_name: Name of the video file
            is_training: Whether this is a training video
            
        Returns:
            Full path to the video file
        """
        base_dir = self.training_dir if is_training else self.test_dir
        return base_dir / f"user_{user_id}" / video_name