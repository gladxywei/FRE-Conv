import copy
import logging
import sys

__all__ = ["get_logger", "set_basic_config"]


global first_set_of_log_config
first_set_of_log_config = False


def get_logger(level=logging.INFO, log_file=None):
    if not first_set_of_log_config:
        set_basic_config(level)
    black, red, green, yellow, blue, magenta, cyan, white = range(8)

    # The background is set with 40 plus the number of the color, and the foreground with 30
    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    def formatter_message(message, use_color=True):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

    COLORS = {
        "WARNING": magenta,
        "INFO": green,
        "DEBUG": blue,
        "CRITICAL": yellow,
        "ERROR": red,
    }

    class ColoredFormatter(logging.Formatter):
        def __init__(self, msg, use_color=True):
            logging.Formatter.__init__(self, msg, datefmt="%Y-%m-%d %H:%M:%S")
            self.use_color = use_color

        def format(self, record):
            record = copy.copy(record)
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    hyde_logger = logging.getLogger("HyDe")
    hyde_stream = logging.StreamHandler()
    stream_supports_color = getattr(hyde_stream.stream, "isatty", lambda: False)()

    hyde_logger.setLevel(level)

    FORMAT = (
        "%(asctime)s - [ $BOLD%(name)-2s$RESET ][ %(levelname)-s ] %(message)s "
        "\t ($BOLD%(filename)s$RESET:%(lineno)d)"
    )
    COLOR_FORMAT = formatter_message(FORMAT, stream_supports_color)

    color_formatter = ColoredFormatter(COLOR_FORMAT, stream_supports_color)
    hyde_stream.setFormatter(color_formatter)

    if log_file is None:
        hyde_logger.handlers = []
        hyde_logger.addHandler(hyde_stream)
    
    else:
        mode = 'a' if hyde_logger.handlers else 'w'
        fhan = logging.FileHandler(log_file,mode=mode)
        fhan.setLevel(logging.DEBUG)


        fhan_FORMAT = formatter_message(FORMAT, False)
        formatter = ColoredFormatter(fhan_FORMAT, False)

        fhan.setFormatter(formatter)

        hyde_logger.handlers = []
        hyde_logger.addHandler(fhan)
        hyde_logger.addHandler(hyde_stream)


    hyde_logger.propagate = False

    return hyde_logger


def set_basic_config(level):
    global first_set_of_log_config
    first_set_of_log_config = True
    
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
