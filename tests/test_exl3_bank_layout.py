"""Algebraic ownership proof for the grouped kernel's shared-memory permutation (no CUDA)."""
import pytest


def swizzle(row, col):
    return col ^ ((row & 1) | ((row & 6) << 2))


@pytest.mark.parametrize('nt', [4,8])
@pytest.mark.parametrize('rows', [4,8,16])
def test_mma_stores_and_row_major_loads_are_bijective_and_conflict_free(nt, rows):
    width = nt * 16
    for row in range(rows):
        assert sorted(swizzle(row,col) for col in range(width)) == list(range(width))
        for begin in range(0,width,32):
            banks=[(row*width+swizzle(row,col))%32 for col in range(begin,begin+32)]
            assert len(set(banks)) == 32
    addresses=[]
    for i in range(nt):
        for h in range(2):
            for c in range(4):
                stores=[]
                for lane in range(32):
                    row=lane//4+(8 if c>=2 else 0)
                    col=i*16+h*8+2*(lane%4)+(c%2)
                    if row<rows:
                        stores.append(row*width+swizzle(row,col))
                assert len({a%32 for a in stores}) == len(stores)
                addresses.extend(stores)
    assert sorted(addresses) == list(range(rows*width))
