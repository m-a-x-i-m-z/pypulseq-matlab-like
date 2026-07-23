from copy import copy, deepcopy


def copy_without_id(event, *, deep: bool = False):
    copied = deepcopy(event) if deep else copy(event)
    if hasattr(copied, 'id'):
        delattr(copied, 'id')
    return copied
