"""Sequence-local acceleration caches with explicit value semantics.

Event registration maps a mutable Python event object to IDs in one
``Sequence``'s compressed event libraries.  Those IDs are meaningful only for
that sequence, so the cache must have the same owner and lifetime. Decompressed
blocks are a different kind of derived state: callers must receive independent
values, never the canonical object retained by the cache.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, MutableMapping, Optional, Tuple

import numpy as np


RegistrationKey = Tuple[Any, ...]


@dataclass(frozen=True)
class _CacheEntry:
    """Latest registration result for one live event object."""

    event: object
    key: RegistrationKey
    registration: Dict[str, Any]


class EventRegistrationCache:
    """Cache event registrations without attaching state to event objects.

    Entries are keyed by ``id(event)`` for fast lookup, but also retain and
    identity-check the event itself.  The strong reference prevents Python
    from reusing the object ID while an entry exists.  Keeping only the latest
    entry for each object bounds growth when callers mutate and reuse events.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = bool(enabled)
        self._entries: Dict[int, _CacheEntry] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        enabled = bool(value)
        if not enabled:
            self.clear()
        self._enabled = enabled

    def get(self, event: object, key: RegistrationKey) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None

        entry = self._entries.get(id(event))
        if entry is None or entry.event is not event or entry.key != key:
            return None
        return entry.registration

    def put(self, event: object, key: RegistrationKey, **registration: Any) -> None:
        if not self._enabled:
            return

        self._entries[id(event)] = _CacheEntry(
            event=event,
            key=key,
            registration=dict(registration),
        )

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __deepcopy__(self, memo) -> 'EventRegistrationCache':
        # Registration IDs are derived state tied to the original Sequence's
        # libraries.  A copied Sequence must rebuild this cache on demand.
        copied = type(self)(enabled=self.enabled)
        memo[id(self)] = copied
        return copied


class BlockCache(MutableMapping[int, object]):
    """Cache decompressed block snapshots without exposing mutable entries.

    Values are deep-copied both when stored and when retrieved. This gives
    cached ``get_block`` calls the same independent-value behavior as MATLAB
    structs. Disabling or switching the cache mode discards all derived state.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = bool(enabled)
        self._entries: Dict[int, object] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._enabled:
            self.clear()
        self._enabled = enabled

    def __getitem__(self, block_id: int) -> object:
        return deepcopy(self._entries[block_id])

    def __setitem__(self, block_id: int, block: object) -> None:
        if self._enabled:
            self._entries[block_id] = deepcopy(block)

    def __delitem__(self, block_id: int) -> None:
        del self._entries[block_id]

    def __iter__(self) -> Iterator[int]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def discard(self, block_id: int) -> None:
        """Invalidate one block without copying the snapshot being discarded."""
        self._entries.pop(block_id, None)

    def __deepcopy__(self, memo) -> 'BlockCache':
        # A copied Sequence rebuilds decompressed blocks from its copied event
        # libraries, avoiding a potentially very large and redundant copy.
        copied = type(self)(enabled=self.enabled)
        memo[id(self)] = copied
        return copied


def make_registration_key(event: object, *field_names: str) -> RegistrationKey:
    """Build a stable key from every event field used during registration."""
    return tuple((field_name, _normalize_key_value(getattr(event, field_name, None))) for field_name in field_names)


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        if contiguous.dtype.hasobject:
            payload = tuple(_normalize_key_value(item) for item in contiguous.flat)
        else:
            payload = contiguous.tobytes()
        return ('ndarray', contiguous.dtype.str, contiguous.shape, payload)
    if isinstance(value, np.generic):
        return _normalize_key_value(value.item())
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, complex):
        return ('complex', value.real.hex(), value.imag.hex())
    if isinstance(value, list):
        return ('list', tuple(_normalize_key_value(item) for item in value))
    if isinstance(value, tuple):
        return ('tuple', tuple(_normalize_key_value(item) for item in value))
    if isinstance(value, dict):
        normalized_items = [
            (_normalize_key_value(key), _normalize_key_value(item))
            for key, item in value.items()
        ]
        return (
            'dict',
            tuple(sorted(normalized_items, key=repr)),
        )
    return value
