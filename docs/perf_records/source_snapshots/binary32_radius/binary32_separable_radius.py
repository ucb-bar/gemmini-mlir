"""Explicit private source-radius precision experiment; no policy selection."""

def prepare_binary32_separable_radius(text: str) -> str:
    declaration = ' merlin_fma_separable_radius radius_plan=merlin_fma_separable_radius_prepare(&product_row,&exact_columns,&anp,&aep,used);'
    original = 'merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j])'
    if text.count(declaration) != 1 or text.count(original) != 1:
        raise ValueError('private source-radius producer/consumer grammar changed')
    text=text.replace(declaration,declaration+'\n merlin_binary32_separable_radius binary32_radius=merlin_binary32_radius_prepare(&radius_plan);')
    position=text.index('#include ')
    text=text[:position]+'#include "binary32_separable_radius.h"\n'+text[position:]
    return text.replace(original,'(binary32_radius.valid?merlin_binary32_radius_apply(&binary32_radius,j,&lo[r*n+j],&hi[r*n+j]):'+original+')')
