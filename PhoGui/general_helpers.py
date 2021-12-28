"""Module dedicated to custom VTK widgets."""

from typing import List, Optional, OrderedDict # for OrderedMeta


# def print_class_helper(obj):
#     return f"<{obj.__class__.__name__}: {obj.__dict__};>"

class SimplePrintable:
    """ Adds the default print method for classes that displays the class name and its dictionary. """
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"


class PrettyPrintable:
    
    def keys(self) -> List[Optional[str]]:
        return self.__dict__.keys()

    def _ipython_key_completions_(self) -> List[Optional[str]]:
        return self.keys()

    # def __setitem__(self, index: Union[Tuple[int, Optional[str]], int, str], data: DataSet):
    #     """Set a block with a VTK data object.

    #     To set the name simultaneously, pass a string name as the 2nd index.

    #     Example
    #     -------
    #     >>> import pyvista
    #     >>> multi = pyvista.MultiBlock()
    #     >>> multi[0] = pyvista.PolyData()
    #     >>> multi[1, 'foo'] = pyvista.UnstructuredGrid()
    #     >>> multi['bar'] = pyvista.PolyData()
    #     >>> multi.n_blocks
    #     3

    #     """
    #     i: int = 0
    #     name: Optional[str] = None
    #     if isinstance(index, (np.ndarray, collections.abc.Sequence)) and not isinstance(index, str):
    #         i, name = index[0], index[1]
    #     elif isinstance(index, str):
    #         try:
    #             i = self.get_index_by_name(index)
    #         except KeyError:
    #             i = -1
    #         name = index
    #     else:
    #         i, name = cast(int, index), None
    #     if data is not None and not is_pyvista_dataset(data):
    #         data = wrap(data)
    #     if i == -1:
    #         self.append(data)
    #         i = self.n_blocks - 1
    #     else:
    #         self.SetBlock(i, data)
    #     if name is None:
    #         name = f'Block-{i:02}'
    #     self.set_block_name(i, name) # Note that this calls self.Modified()
    #     if data not in self.refs:
    #         self.refs.append(data)

    # def __delitem__(self, index: Union[int, str]):
    #     """Remove a block at the specified index."""
    #     if isinstance(index, str):
    #         index = self.get_index_by_name(index)
    #     self.RemoveBlock(index)

    # def __iter__(self) -> 'MultiBlock':
    #     """Return the iterator across all blocks."""
    #     self._iter_n = 0
    #     return self

    # def next(self) -> Optional['MultiBlock']:
    #     """Get the next block from the iterator."""
    #     if self._iter_n < self.n_blocks:
    #         result = self[self._iter_n]
    #         self._iter_n += 1
    #         return result
    #     else:
    #         raise StopIteration

    # __next__ = next
    
    def _repr_pretty_(self, p, cycle=False):
        """ The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself. """
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.__dict__.__repr__() if not cycle else '...')
        # return self.as_array().__repr__() # p.text(repr(self))



class OrderedMeta(type):
    """ Replaces the inheriting object's dict of attributes with an OrderedDict that preserves enumeration order
    Reference: https://stackoverflow.com/questions/11296010/iterate-through-class-members-in-order-of-their-declaration
    Usage:
        # Set the metaclass property of your custom class to OrderedMeta
        class Person(metaclass=OrderedMeta):
            name = None
            date_of_birth = None
            nationality = None
            gender = None
            address = None
            comment = None
    
        # Can then enumerate members while preserving order
        for member in Person._orderedKeys:
            if not getattr(Person, member):
                print(member)
    """
    @classmethod
    def __prepare__(metacls, name, bases): 
        return OrderedDict()

    def __new__(cls, name, bases, clsdict):
        c = type.__new__(cls, name, bases, clsdict)
        c._orderedKeys = clsdict.keys()
        return c

