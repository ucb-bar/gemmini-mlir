"""Combined static WS reuse policy, independent of program identities."""


def _b_resident_eligible(M,K,N):
    """Match weightreuse admission and prove its padded local footprints.

    The retained panel has four M subtiles, at most four N subtiles, and all K
    subtiles. A uses four padded K rows; B uses 64 rows for each K subtile.
    """
    if any(type(v) is not int for v in (M,K,N)):
        return False
    if not (M>64 and 0<K<=2048 and N>=16):
        return False
    ktiles=(K+15)//16
    kpad=ktiles*16
    a_rows=4*kpad
    b_rows=ktiles*64
    acc_rows=3*64+((min(N,64)+15)//16)*16
    return a_rows<=8192 and b_rows<=8192 and acc_rows<=256


def emit_static_product(a,b,y,M,K,N,width,scale,relu,bias,tag):
    if _b_resident_eligible(M,K,N):
        from device_static_weightreuse import emit_static_product as emit
    else:
        from device_static_panelreuse import emit_static_product as emit
    return emit(a,b,y,M,K,N,width,scale,relu,bias,tag)
