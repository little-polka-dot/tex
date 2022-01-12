import logging


class TexLogger(logging.Logger):

    __logger_name__ = 'tex'
    __console_format__ = '[%(levelname)s] [%(asctime)s] %(message)s'
    __console_level__ = 'INFO'

    @classmethod
    def logger(cls) -> logging.Logger:
        if not hasattr(cls, '__logger__'):
            setattr(cls, '__logger__', cls())
        return getattr(cls, '__logger__')

    def __init__(self, **kwargs):
        super().__init__(self.__logger_name__, **kwargs)
        self.console = logging.StreamHandler()
        self.console.setFormatter(
            logging.Formatter(self.__console_format__))
        self.console.setLevel(self.__console_level__)
        self.addHandler(self.console)


if __name__ == '__main__':
    TexLogger.logger().info('test message')
