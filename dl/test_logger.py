from pathlib import Path
import logging



def foo(path):
    logging.basicConfig(level=logging.INFO, filename=path.joinpath('info.log'))
    logging.info("hi")
