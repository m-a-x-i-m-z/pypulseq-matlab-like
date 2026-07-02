from types import SimpleNamespace
from typing import Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar('Self', bound='EventLibrary')

import math
import numbers

import numpy as np


class EventLibrary:
    """
    Defines an event library to maintain a list of events. Provides methods to insert new data and find existing data.

    Sequence Properties:
    - data - A struct array with field 'array' to store data of varying lengths, remaining compatible with codegen.
    - type - Type to distinguish events in the same class (e.g. trapezoids and arbitrary gradients)

    Sequence Methods:
    - find - Find an event in the library
    - insert - Add a new event to the library

    See also `Sequence.py`.

    Attributes
    ----------
    data : dict{str: numpy.array}
        Key-value pairs of event keys and corresponding data.
    type : dict{str, str}
        Key-value pairs of event keys and corresponding event types.
    keymap : dict{str, int}
        Key-value pairs of data values and corresponding event keys.
    """

    def __init__(self, numpy_data=False):
        self.keys = {}
        self.data = {}
        self.lengths = {}
        self.type = {}
        self.keymap = {}
        self.lookup_key = lambda key, fallback=0: self.keymap.get(key, fallback)
        self.next_free_ID = 1
        self.numpy_data = numpy_data

    def _key_from_data(self, new_data: np.ndarray | list) -> str:
        """
        Build a key for event lookup. For numpy_data libraries, match MATLAB's
        sprintf('%.6g ', data) behavior to align duplicate detection.
        """
        if self.numpy_data:
            data = np.asarray(new_data).ravel()
            return ' '.join(f'{x:.6g}' if isinstance(x, numbers.Number) else str(x) for x in data)
        return tuple(new_data)

    def __str__(self) -> str:
        s = 'EventLibrary:'
        s += '\ndata: ' + str(len(self.data))
        s += '\ntype: ' + str(len(self.type))
        return s

    def find(self, new_data: np.ndarray) -> Tuple[int, bool]:
        """
        Finds data `new_data` in event library.

        Parameters
        ----------
        new_data : numpy.ndarray
            Data to be found in event library.

        Returns
        -------
        key_id : int
            Key of `new_data` in event library, if found.
        found : bool
            If `new_data` was found in the event library or not.
        """
        key = self._key_from_data(new_data)

        key_id = self.lookup_key(key, 0)
        if key_id != 0:
            found = True
        else:
            key_id = self.next_free_ID
            found = False

        return key_id, found

    def find_or_insert(self, new_data: np.ndarray, data_type: str = str()) -> Tuple[int, bool]:
        """
        Lookup a data structure in the given library and return the index of the data in the library. If the data does
        not exist in the library it is inserted right away. The data is a 1xN array with event-specific data.

        See also  insert `pypulseq.Sequence.sequence.Sequence.add_block()`.

        Parameters
        ----------
        new_data : numpy.ndarray
            Data to be found (or added, if not found) in event library.
        data_type : str, default=str()
            Type of data.

        Returns
        -------
        key_id : int
            Key of `new_data` in event library, if found.
        found : bool
            If `new_data` was found in the event library or not.
        """
        if self.numpy_data:
            new_data = np.asarray(new_data)
            new_data.flags.writeable = False
        key = self._key_from_data(new_data)

        key_id = self.lookup_key(key, 0)
        if key_id != 0:
            found = True
        else:
            key_id = self.next_free_ID
            found = False

            # Insert
            self.data[key_id] = new_data
            self.keys[key_id] = key_id
            self.lengths[key_id] = len(new_data)

            if data_type != str():
                self.type[key_id] = data_type

            self.keymap[key] = key_id
            self.next_free_ID = key_id + 1  # Update next_free_id

        return key_id, found

    def insert(self, key_id: int, new_data: np.ndarray | list, data_type: str = str()) -> int:
        """
        Add event to library.

        See also `pypulseq.event_library.EventLibrary.find()`.

        Parameters
        ----------
        key_id : int
            Key of `new_data`.
        new_data : numpy.ndarray or list
            Data to be inserted into event library.
        data_type : str, default=str()
            Data type of `new_data`.

        Returns
        -------
        key_id : int
            Key ID of inserted event.
        """
        if isinstance(key_id, float):
            key_id = int(key_id)

        if key_id == 0:
            key_id = self.next_free_ID

        if self.numpy_data:
            new_data = np.asarray(new_data)
            new_data.flags.writeable = False
        key = self._key_from_data(new_data)

        self.data[key_id] = new_data
        self.keys[key_id] = key_id
        self.lengths[key_id] = len(new_data)
        if data_type != str():
            self.type[key_id] = data_type

        self.keymap[key] = key_id

        if key_id >= self.next_free_ID:
            self.next_free_ID = key_id + 1  # Update next_free_id

        return key_id

    def get(self, key_id: int) -> dict:
        """

        Parameters
        ----------
        key_id : int

        Returns
        -------
        dict
        """
        return {
            'key': self.keys.get(key_id, key_id),
            'data': self.data[key_id],
            'length': self.lengths.get(key_id, len(self.data[key_id])),
            'type': self.type.get(key_id, str()),
        }

    def out(self, key_id: int) -> SimpleNamespace:
        """
        Get element from library by key.

        See also `pypulseq.event_library.EventLibrary.find()`.

        Parameters
        ----------
        key_id : int

        Returns
        -------
        out : SimpleNamespace
        """
        out = SimpleNamespace()
        out.key = self.keys.get(key_id, key_id)
        out.data = self.data[key_id]
        out.length = self.lengths.get(key_id, len(self.data[key_id]))
        out.type = self.type.get(key_id, str())

        return out

    def update(
        self,
        key_id: int,
        old_data: Union[np.ndarray, None],
        new_data: np.ndarray,
        data_type: str = str(),
    ):
        """
        Parameters
        ----------
        key_id : int
        old_data : numpy.ndarray (Ignored!)
        new_data : numpy.ndarray
        data_type : str, default=str()
        """
        if old_data is not None:
            old_key = self._key_from_data(old_data)
        elif key_id in self.data:
            old_key = self._key_from_data(self.data[key_id])
        else:
            old_key = None

        if old_key is not None and old_key in self.keymap:
            del self.keymap[old_key]

        self.insert(key_id, new_data, data_type)

    def update_data(
        self,
        key_id: int,
        old_data: np.ndarray,
        new_data: np.ndarray,
        data_type: str = str(),
    ):
        """
        Parameters
        ----------
        key_id : int
        old_data : np.ndarray (Ignored!)
        new_data : np.ndarray
        data_type : str
        """
        self.update(key_id, old_data, new_data, data_type)

    def remove_duplicates(self, digits: Union[int, Tuple[int]]) -> Tuple[Self, dict]:
        """
        Remove duplicate events from this event library by rounding the data
        according to the significant `digits` specification, and then removing
        duplicate events.
        Returns a new event library, leaving the current one intact.

        Parameters
        ----------
        digits : Union[int, List[int]]
            For libraries with `numpy_data == True`:
                A single number specifying the number of significant digits
                after rounding.
            Otherwise:
                A tuple of numbers specifying the number of significant digits
                after rounding for each entry in the event data tuple.

        Returns
        -------
        new_library : EventLibrary
            Event library with the duplicate events removed
        mapping : dict
            Dictionary containing a mapping of IDs in the old library to IDs
            in the new library.
        """

        def round_value(value, digit):
            if not isinstance(value, numbers.Number):
                return value
            return round(value, digit - math.ceil(math.log10(abs(value) + 1e-12)) if digit > 0 else -digit)

        def round_data(data: Tuple[float], digits: Tuple[int]) -> Tuple[float]:
            """
            Round the data tuple to a specified number of significant digits,
            specified by `digits`. Rounding behavior is similar to the {.Ng}
            format specifier if N > 0, and similar to {.0f} otherwise.
            """
            rounded = [round_value(d, dig) for d, dig in zip(data, digits, strict=False)]
            if len(data) > len(rounded):
                rounded.extend(data[len(rounded):])
            return tuple(rounded)

        def round_data_numpy(data: np.ndarray, digits: Union[int, Tuple[int]]) -> np.ndarray:
            """
            Round the data array to a specified number of significant digits,
            specified by `digits`. Rounding behavior is similar to the {.Ng}
            format specifier if N > 0, and similar to {.0f} otherwise.
            """
            if isinstance(digits, (tuple, list)):
                # Handle element-wise rounding if digits is a tuple/list
                # Only use as many digits as there are data points (matching strict=False behavior)
                n = min(len(data), len(digits))
                result_data = np.array(
                    [round_value(d, dig) for d, dig in zip(data[:n], digits[:n], strict=False)],
                    dtype=object if data.dtype.kind in 'OUS' else data.dtype,
                )
                if len(data) > n:
                    result = np.concatenate([result_data, data[n:]])
                else:
                    result = result_data
            else:
                mags = 10 ** (digits - (np.ceil(np.log10(abs(data) + 1e-12))) if digits > 0 else -digits)
                result = np.round(data * mags) / mags
            
            result.flags.writeable = False
            return result

        # Round library data based on `digits` specification
        if self.numpy_data:
            rounded_data = {x: round_data_numpy(self.data[x], digits) for x in self.data}
        else:
            rounded_data = {x: round_data(self.data[x], digits) for x in self.data}

        # Initialize filtered library
        new_library = EventLibrary(numpy_data=self.numpy_data)

        # Initialize ID mapping. Always include 0:0 to allow the mapping dict
        # to be used for mapping block_events (which can contain 0, i.e. no
        # event)
        mapping = {0: 0}

        # Recreate library using rounded values
        for k, v in sorted(rounded_data.items()):
            mapping[k], _ = new_library.find_or_insert(v, self.type[k] if k in self.type else str())

        return new_library, mapping
