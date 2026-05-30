from src.pipeline.training_pipeline import TrainingPipeline

from src.logger import get_logger

logger = get_logger(__name__)
if __name__ == "__main__":

    try:
        logger.info("PatrolIQ Training Started")

        pipeline = TrainingPipeline()
        pipeline.run()

        logger.info("PatrolIQ Training Finished")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        raise
