import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/twhuang/experiment/new_vistadream/VistaDream/logger.txt'),
        # 2: 4 all panora
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)