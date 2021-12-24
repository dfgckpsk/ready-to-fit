from ..data.MlData import MlData


class CVBase:

    def split(self, mldata: MlData):

        """
        Split generator
        :param mldata:
        :return:
        """

    def __str__(self):
        not_private_keys = list(filter(lambda x: x[0] != '_', self.__dict__.keys()))
        attr_dict = { your_key: self.__dict__[your_key] for your_key in not_private_keys }
        return f'{self.__class__.__name__}({attr_dict})'
