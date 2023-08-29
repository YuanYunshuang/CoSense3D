import logging


class ProcessNode(object):
    def __init__(self, **kwargs):
        logging.info(f"{self.__class__.__name__}:")
        for k, v in kwargs.items():
            setattr(self, k, v)
            logging.info(f"- {k}: {v}")

    def __call__(self, data_dict):
        raise NotImplementedError