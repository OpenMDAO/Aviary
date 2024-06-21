import itertools
import numpy as np


def _rep(n, t):
    """Shorthand for ``itertools.repeat`` with the multiplier first."""
    return itertools.repeat(t, n)


def _parse(f, fmt):
    """Read a line from file ``f`` and parse it according to the given ``fmt``"""
    return _strparse(f.readline(), fmt)


def _strparse(s, fmt):
    """Parse a string into fixed-width numeric fields.
    ``fmt`` should be a list of tuples specifying (type, length) for each field in
    string ``s``. Use None for the type to skip (i.e. not yield) that field.
    """
    p = 0
    for typ, length in fmt:
        sub = s[p: p + length]
        if typ is not None:
            yield typ(sub)
        p += length


def _read_map(f, is_turbo_prop=False):
    """Read a single map of a table from the engine deck or propeller map file.
    The map data is returned in the same format as in ``read_table`` except
    there is a single altitude value per map in the case of engine deck and
    there is a single Mach number per map in the case of propeller map.
    """
    # map dimensions: FORMAT(/2I5,F10.1,10X))
    npts, nline, amap = _parse(f, [*_rep(2, (int, 5)), (float, 10)])

    map_data = np.empty((npts * nline, 4))
    map_data[:, 0] = amap

    # number of points on a single line - wrapped if more than 6
    max_columns = 6

    # point vals: FORMAT(10X,6F10.4,10X)
    x = []
    npts_remaining = npts
    while npts_remaining > 0:
        npts_to_read = min(max_columns, npts_remaining)
        # remaining vals on wrapped line
        x.extend(list(_parse(f, [(None, 10), *_rep(npts_to_read, (float, 10))])))
        npts_remaining -= npts_to_read

    map_data[:, 2] = np.tile(x, nline)

    for j in range(nline):
        npts_remaining = npts
        npts_to_read = min(max_columns, npts_remaining)
        # line (y) val then z vals: FORMAT(F10.4,6F10.1,10X,/(6F10.1,10X))
        vals = list(_parse(f, [(float, 10), *_rep(npts_to_read, (float, 10))]))
        y = vals[0]
        z = vals[1:]
        npts_remaining -= npts_to_read
        while npts_remaining > 0:
            npts_to_read = min(max_columns, npts_remaining)
            # add remaining vals on warapped line
            line_format = [*_rep(npts_to_read, (float, 10))]
            if is_turbo_prop:
                line_format = [(None, 10), *line_format]
            z.extend(list(_parse(f, line_format)))
            npts_remaining -= npts_to_read

        sl = slice(j * npts, (j + 1) * npts)
        map_data[sl, 1] = y
        map_data[sl, 3] = z

    return map_data
