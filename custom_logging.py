from termcolor import colored
import logging
CHAIN1 = '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'
CHAIN2 = '____________________________________________________________________________________________________'
CHAIN3 = '****************************************************************************************************'
NEWLINE = '\n\t\t\t     '
import inspect
import os

def create_logger(app_name=None):
    logger = logging.getLogger(app_name or __name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    return logger

LOGGER = create_logger()

class Blogger():
    def __init__(self, display_caller = False):
        self.display_caller = display_caller # Whether to log caller function for easier debugging

    @staticmethod
    def _get_fmt_caller():
        (filename, line_number,
        caller_name, lines, index) =  inspect.getframeinfo(inspect.currentframe().f_back.f_back) # Back two steps - from _fmt_caller and the log function
        fmt_caller = " ({})".format(colored(caller_name, 'red'))
        return fmt_caller

    def log(self, msg):
        msg = colored(msg, 'white')
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            LOGGER.info("{}   {}".format(msg, fmt_caller))
        else:
            LOGGER.info(msg)

    def green(self, msg):
        msg = colored(msg, 'green')
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            LOGGER.info("{}   {}".format(msg, fmt_caller))
        else:
            LOGGER.info(msg)

    def yellow(self, msg):
        msg = colored(msg, 'yellow')
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            LOGGER.info("{}   {}".format(msg, fmt_caller))
        else:
            LOGGER.info(msg)

    def red(self, msg):
        msg = colored(msg, 'red')
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            LOGGER.error("{}   {}".format(msg, fmt_caller))
        else:
            LOGGER.error(msg)

    def status_update(self, msg):
        msg = colored(msg, 'cyan')
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            LOGGER.info("{}   {}".format(msg, fmt_caller))
        else:
            LOGGER.info(msg)

    def begin_status_update(self, msg):
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            msg_with_caller = "{}   {}".format(colored(msg, 'green'), fmt_caller)
            first_half = colored(CHAIN1 + NEWLINE, 'green')
            second_half = colored(NEWLINE + CHAIN1, 'green')
            LOGGER.info("{} {} {}".format(first_half, msg_with_caller, second_half))
        else:
            msg = colored(CHAIN1 + NEWLINE + msg + NEWLINE + CHAIN1, 'green')
            LOGGER.info(msg)

    def end_status_update(self, msg):
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            msg_with_caller = "{}   {}".format(colored(msg, 'green'), fmt_caller)
            first_half = colored(CHAIN2 + NEWLINE, 'green')
            second_half = colored(NEWLINE + CHAIN2, 'green')
            LOGGER.info("{} {} {}".format(first_half, msg_with_caller, second_half))
        else:
            msg = colored(CHAIN2 + NEWLINE + msg + NEWLINE + CHAIN2, 'green')
            LOGGER.info(msg)

    def queue_full_log(self, msg):
        if self.display_caller:
            fmt_caller = self._get_fmt_caller()
            msg_with_caller = "{}   {}".format(colored(msg, 'yellow'), fmt_caller)
            first_half = colored(CHAIN3 + NEWLINE, 'yellow')
            second_half = colored(NEWLINE + CHAIN3, 'yellow')
            LOGGER.info("{} {} {}".format(first_half, msg_with_caller, second_half))
        else:
            msg = colored(CHAIN3 + NEWLINE + msg + NEWLINE + CHAIN3, 'yellow')
            LOGGER.info(msg)
