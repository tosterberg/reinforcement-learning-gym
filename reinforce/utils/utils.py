"""
    Utility Functions
        conversions:
            dict -> str (recursive)
"""


def dict_to_string(obj):
    out_str = '{'
    for k, v in obj.__dict__.items():
        if hasattr(v, '__iter__') and not isinstance(v, str):
            v = ", ".join([str(x) for x in v])
        out_str += f'{str(k)}: {str(v)}, '
    out_str = out_str[:-2] + '}'
    return out_str
