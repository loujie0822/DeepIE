import logging


class Logger(object):
    def __init__(self, logfile=None):
        self.logfile = logfile
        self.formats = '%(asctime)s:[%(filename)s:%(lineno)d] - %(message)s'
        self.formatter = logging.Formatter(self.formats)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, *args, **kwargs):
        return self.logger

    def set_log_file(self, logfile, console=False):
        self.logfile = logfile
        self.logger_to_file(console)

    def logger_to_file(self, console):
        #logging.basicConfig(level=logging.INFO, format=formats, filename=logfile)
        # 写入控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter)
        if console:
            self.logger.addHandler(ch)

        # 写入文件
        fh = logging.FileHandler(self.logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)


logger_ins = Logger()
logger = logger_ins()
