import logging

from config import DEBUG, WRITE_CSV


class BaseDataHandler:

    logger = logging.getLogger(__name__)
    debug_ = DEBUG
    write_csv = WRITE_CSV
