from loguru import logger
import sys

logger.remove()

fmt = "<b><g>{time:MM-DD HH:mm:ss}</g> | <lvl>{level:<8}</lvl> | <e>{file}:{line:<4}</e> | <lvl>{message}</lvl></b>"

logger.add(sys.stdout, format=fmt)