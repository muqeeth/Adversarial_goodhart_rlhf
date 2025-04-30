import logging
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=1)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state)
accelerator.state.deepspeed_plugin