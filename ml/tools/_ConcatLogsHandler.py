import logging


class _ConcatLogsHandler(logging.Handler):

    def __init__(self):
        super().__init__()
        self.concated_logs = ''

    def emit(self, record):
        msg = self.format(record)
        self.concated_logs += msg + '\n'
