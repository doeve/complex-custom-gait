# File: examples/basic_usage.py

from pathlib import Path
from src.models.recognition_system import GaitRecognitionSystem
from loguru import logger

def main():
    # Initialize the system
    system = GaitRecognitionSystem("configs/model_config.yaml")

    # # Add a new user
    # user_name = "John Doe"
    # user_id = system.add_user(user_name)
    # logger.info(f"Added new user: {user_name} with ID: {user_id}")

    # # Add training videos for the user
    # training_videos = [
    #     "data/sample_videos/john_walking_1.mp4",
    #     "data/sample_videos/john_walking_2.mp4"
    # ]
    
    # for video_path in training_videos:
    #     system.add_training_video(user_id, video_path)
    #     logger.info(f"Added training video: {video_path}")

    # Train the model for this user
    logger.info("Training model for user...")
    system.train_user(1)
    logger.info("Training complete for user 1")
    system.train_user(2)
    logger.info("Training complete for user 2")

    # Test identification
    test_video = "data/sample_videos/tibi_test.mp4"
    identified_id, confidence = system.identify_person(test_video)

    if identified_id is not None:
        user = system.users[identified_id]
        logger.info(f"Identified person: {user.name} with confidence: {confidence:.2f}")
    else:
        logger.info(f"No person identified (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()