from logging import handlers
import logging
import time


__Author__ = 'Quanhao Guo'


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }# Log level relationship mapping

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)# Set log format
        self.logger.setLevel(self.level_relations.get(level))# Set log level
        sh = logging.StreamHandler()# Output to the screen
        sh.setFormatter(format_str) # Set the format displayed on the screen
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#Write to the file '#' The processor that automatically generates the file at the specified interval
        # Instantiate TimedRotatingFileHandler
         # interval is the time interval, backupCount is the number of backup files, if it exceeds this number, it will be automatically deleted, when is the time unit of the interval, the units are as follows:
         # S seconds
         # M points
         # H hours,
         # D day,
         # W Every week (interval==0 means Monday)
         # midnight Every morning
        th.setFormatter(format_str)# Set the format written in the file
        self.logger.addHandler(sh) # Add object to logger
        self.logger.addHandler(th)


if __name__ == '__main__':
    time_now = time.strftime("%Y%m%d-%H.%M", time.localtime())
    log = Logger(time_now+'.log',level='info')
    log.logger.info('info')
