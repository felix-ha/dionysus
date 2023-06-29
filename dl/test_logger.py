import logging



def foo():
    logging.basicConfig(level=logging.INFO, filename='info.log')
    logging.info("hi")
