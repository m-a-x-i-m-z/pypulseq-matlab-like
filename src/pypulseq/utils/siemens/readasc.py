import os
import re
from typing import Tuple


def readasc(filename: str) -> Tuple[dict, dict]:
    """
    Reads Siemens ASC ascii-formatted textfile and returns a dictionary
    structure.
    E.g. a[0].b[2][3].c = "string"
    parses into:
      asc['a'][0]['b'][2][3]['c'] = "string"

    Parameters
    ----------
    filename : str
        Filename of the ASC file.

    Returns
    -------
    asc : dict
        Dictionary of ASC part of file.
    extra : dict
        Dictionary of other fields after "ASCCONV END"
    """
    def _set_nested(base: dict, field_name: str, value):
        tokens = []
        for part in field_name.split('.'):
            m = re.match(r'^([A-Za-z0-9_]+)', part)
            if m is None:
                raise RuntimeError(f'Invalid ASC field segment: {part}')
            tokens.append(m.group(1))
            for idx in re.findall(r'\[(\d+)\]', part):
                tokens.append(int(idx))

        d = base
        for token in tokens[:-1]:
            if token not in d or not isinstance(d[token], dict):
                d[token] = {}
            d = d[token]
        d[tokens[-1]] = value

    def _parse_value(field_name: str, raw_value: str):
        v = raw_value.strip()
        if '"' in v:
            v = v.replace('"', '')
        if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]

        if v.lower().startswith('0x') and 'atImagedNucleus' not in field_name:
            try:
                return int(v[2:], 16)
            except ValueError:
                pass

        if re.fullmatch(r'[+-]?\d+', v):
            return int(v)

        if re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', v):
            return float(v)

        return v

    asc, extra = {}, {}
    base_path, ext = os.path.splitext(filename)
    input_files = [filename]
    safety_file = f'{base_path}_GSWD_SAFETY{ext}'
    if os.path.exists(safety_file):
        input_files.append(safety_file)

    # Read asc file(s) and convert into nested dictionaries.
    for current_file in input_files:
        with open(current_file, 'r', encoding='utf-8', errors='ignore') as fp:
            end_of_asc = False

            for next_line in fp:
                next_line = next_line.strip()

                if next_line == '### ASCCONV END ###':
                    end_of_asc = True
                    continue

                if next_line == '' or next_line[0] == '#':
                    continue

                if '=' not in next_line:
                    continue

                field_name, raw_value = next_line.split('=', 1)
                field_name = field_name.strip()
                raw_value = raw_value.strip()

                # MATLAB readasc strips comments after '#' or '//' on the
                # right-hand side before value conversion.
                comments = [i for i in (raw_value.find('#'), raw_value.find('//')) if i >= 0]
                if comments:
                    raw_value = raw_value[: min(comments)].rstrip()

                if field_name == '':
                    continue

                target = extra if end_of_asc else asc
                _set_nested(target, field_name, _parse_value(field_name, raw_value))

    return asc, extra
