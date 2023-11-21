import sys
import logging
import os

class LoggingFunction :
    def __init__(self, ModuleName) :
        self.path = self.CheckLoggingPath()
        logger = logging.getLogger(ModuleName)

        logger.setLevel(logging.DEBUG)

        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        errHandler = logging.FileHandler(f"{self.path}/{ModuleName}.log")
        
        stdoutHandler.setLevel(logging.DEBUG)
        errHandler.setLevel(logging.ERROR)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s --> %(message)s"
        )

        stdoutHandler.setFormatter(fmt)
        errHandler.setFormatter(fmt)

        logger.addHandler(stdoutHandler)
        logger.addHandler(errHandler)
        self.Logger = logger
        
    def CheckLoggingPath(self) :
        if os.path.exists("logging") :
            pass
        else :
            os.mkdir("logging")
        return "logging"
    
    
    def logInfoMessage(self, message) :
        return self.Logger.info(message)

    def logDebugMessage(self, message) :
        return self.Logger.debug(message)

    def logWarningMessage(self, message) :
        return self.Logger.warning(message)
    
    def logErrorMessage(self, message, status) :
        return self.Logger.error(message, exc_info=status)

    def logCriticalMessage(self, message) :
        return self.Logger.critical(message)
