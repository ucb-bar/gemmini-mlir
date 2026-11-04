"""Compact runtime loops for repeated, explicit column-gain tile patterns."""
from isa_encoder import float_bits


def emit_column_groups(e, groups, callback, loop_repeated=False):
    """Call callback(nSSA, width, gain) in the supplied column-group order.

    Optional compaction loops only over contiguous full 16-column tiles with
    identical relative group starts, widths and f32 gain bits. Partial tiles,
    gaps and nonrepeated patterns retain static iteration. No gains are inferred.
    """
    if not loop_repeated:
        for start, width, gain in groups:
            callback(e.value(start), width, gain)
        return

    # Preserve input order, including unusual partial/gapped tile schedules.
    tiles = []
    for start, width, gain in groups:
        base = start // 16 * 16
        if not tiles or tiles[-1][0] != base:
            tiles.append((base, []))
        tiles[-1][1].append((start, width, gain))

    def pattern(base, entries):
        cursor = base
        result = []
        for start, width, gain in entries:
            if start != cursor or width <= 0 or start + width > base + 16:
                return None
            result.append((start - base, width, float_bits(gain)))
            cursor += width
        return tuple(result) if cursor == base + 16 else None

    patterns = [pattern(base, entries) for base, entries in tiles]
    first = 0
    while first < len(tiles):
        last = first + 1
        if patterns[first] is not None:
            while (last < len(tiles) and tiles[last][0] == tiles[last - 1][0] + 16
                   and patterns[last] == patterns[first]):
                last += 1
        if last - first >= 2:
            base, entries = tiles[first]
            def tile(column_base):
                for start, width, gain in entries:
                    column = e.b.add_i(column_base, e.value(start - base))
                    callback(column, width, gain)
            e.for_loop(base, tiles[last - 1][0] + 16, 16, tile)
        else:
            for start, width, gain in tiles[first][1]:
                callback(e.value(start), width, gain)
        first = last
