# Quick patch to ensure log handler flushes immediately
import logging

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
