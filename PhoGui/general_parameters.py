    
from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable


class VisualizationParameters(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, name) -> None:
        # super(VisualizationParameters, self).__init__(**kwargs)
        self.name = name
        # self.__dict__ = (self.__dict__ | kwargs)
        
    # def __init__(self, name, **kwargs) -> None:
    #     # super(VisualizationParameters, self).__init__(**kwargs)
    #     self.name = name
    #     # self.__dict__ = (self.__dict__ | kwargs)
    

class DebugHelper(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
    # def __init__(self, name, **kwargs) -> None:
    def __init__(self, name) -> None:
        # super(DebugHelper, self).__init__(**kwargs)
        self.name = name
        # self.__dict__ = (self.__dict__ | kwargs)

