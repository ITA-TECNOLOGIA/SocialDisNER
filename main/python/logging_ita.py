import logging ,os


def configure():
    # create logger
    #in windows set LOGLEVEL=DEBUG
    #in linux export LOGLEVEL=DEBUG
    #default  'LOGLEVEL', 'INFO'
    LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
    
    logger = logging.getLogger("logging_ita")
    logger.setLevel(logging.DEBUG)
    
    logger.setLevel(level=LOGLEVEL)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level=LOGLEVEL)

    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(lineno)d;%(filename)s;%(message)s", "%Y-%m-%d %H:%M:%S")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    if not logger.hasHandlers():
        logger.addHandler(ch)

    # "application" code
    logger.debug("debug message")
    logger.info("info message")
    logger.warn("warn message")
    logger.error("error message")
    logger.critical("critical message")
    return logger