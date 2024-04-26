from src.logger import Logger
import logging
logger = Logger('logs')

def test_logger():
    logger.log('Test message')
    # assert os.path.exists('logs/log.txt')
    logger.log('Test message', level=logging.WARNING)

test_logger()

def divide_by_zero():
    try:
        1/0
    except Exception as e:
        logger.log(str(e), level=logging.ERROR)
        print('Error:', e)

divide_by_zero()