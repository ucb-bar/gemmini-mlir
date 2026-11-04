"""Explicit composition of regional affine origins and image transfer runs.

The existing regional policy supplies the source/panel/epoch product. Optional
transfer planning consumes that exact product. Both affect the existing common
emitter; neither changes arithmetic, output ownership, or completion ordering.
This bundle is provisional and does not change any compiler default.
"""
from __future__ import annotations

from dataclasses import dataclass

from .conv_regional import ConvSpatialProduct, plan_conv_regional
from .conv_transfer_runs import ConvTransferRuns, plan_transfer_runs
from .conv_spatial_panels import _emit_prepared

OPTIONS = ('affine_uniform_stores', 'affine_origins', 'all_affine_uniform_stores')


@dataclass(frozen=True)
class ConvTransferRegional:
    regional: ConvSpatialProduct
    transfers: ConvTransferRuns | None
    selection: dict
    option: str = 'affine_uniform_stores'

    @property
    def actual_mode(self):
        if self.option == 'all_affine_uniform_stores':
            return 'affine_uniform_stores'
        if self.regional.trial_mode == 'affine_origins' and self.option == 'affine_uniform_stores':
            return 'affine_uniform_stores'
        return self.regional.trial_mode

    @property
    def route(self):
        mode = self.actual_mode
        return mode + ('_transfer_runs' if self.transfers is not None else '')

    def revalidate(self, ctx):
        if self.option not in OPTIONS:
            raise ValueError('unknown convolution transfer regional option')
        if type(self.regional) is not ConvSpatialProduct:
            raise ValueError('convolution bundle requires the original typed regional product')
        self.regional.revalidate(ctx)
        if self.transfers is not None:
            if (type(self.transfers) is not ConvTransferRuns
                    or self.transfers.prepared is not self.regional.prepared):
                raise ValueError('convolution bundle products differ in prepared source/allocation')
            self.transfers.revalidate(self.regional.prepared)

    def emit(self, ctx):
        """Consume the retained products only after every admission revalidates."""
        self.revalidate(ctx)
        regional = self.regional
        if self.transfers is None and self.actual_mode == regional.trial_mode:
            emitted = regional.emit(ctx)
        else:
            # Compatibility with hoisted origins follows from the same source
            # mapping, not from an extra target execution law. The joined tap
            # computes image/spatial coordinates directly from absolute first;
            # it never interprets a padded gather origin as a physical address.
            # Every other tap calls the original _input_fragments with the
            # unchanged origins supplied by the common emitter. Thus the
            # hoisted and unhoisted paths retain identical per-row source maps.
            emitted = _emit_prepared(ctx, regional.ifm, regional.rhs, regional.dst,
                regional.prepared, trial_mode=self.actual_mode,
                input_fragment_emitter=self.transfers.emit if self.transfers else None)
        prepared = regional.prepared
        if (not emitted or emitted['source_sha256'] != prepared.source.identity
                or emitted['trial_mode'] != self.actual_mode
                or emitted['rhs_rows'] != prepared.group.panel.rhs_rows
                or emitted['rhs_base'] != prepared.group.panel.rhs_base):
            raise ValueError('convolution bundle emitted a different selected product')
        receipt = dict(schema='authored_conv_transfer_regional.v1', route=self.route,
            qualification='explicit provisional composition; no automatic default promotion',
            mode_authority='explicit all-affine ablation' if self.option == 'all_affine_uniform_stores'
                else 'existing regional policy plus explicit uniform-store composition'
                if self.option == 'affine_uniform_stores' else 'existing regional policy',
            source_sha256=prepared.source.identity, option=self.option,
            selected_regional_mode=regional.trial_mode, actual_emission_mode=self.actual_mode,
            regional_selection=self.selection,
            regional_binding_sha256=regional.binding_sha256,
            transfer_product=self.transfers.receipt() if self.transfers else None,
            origin_compatibility='joined tap uses absolute first; other taps retain original hoisted origins',
            completion='original shared full-K/N epoch and output/phase last-use fences',
            emitted=emitted)
        ctx.metadata.setdefault('convolution_transfer_regional', []).append(receipt)
        return receipt


def plan_transfer_regional(ctx, ifm, rhs, dst, attributes, *, option='affine_uniform_stores'):
    """Read-only composition of existing typed products; no new shape selector."""
    if option not in OPTIONS:
        raise ValueError('convolution transfer regional option must be one of '+repr(OPTIONS))
    regional, selection = plan_conv_regional(ctx, ifm, rhs, dst, attributes,
        option='provisional_affine_fragmented', allow_wide=True)
    if regional is None:
        return None
    transfers = plan_transfer_runs(regional.prepared)
    product = ConvTransferRegional(regional, transfers, selection, option)
    product.revalidate(ctx)
    return product


def lower_transfer_regional(ctx, ifm, rhs, dst, attributes, *, option='affine_uniform_stores'):
    """Emit the optional composed route, or None before mutation if refused."""
    product = plan_transfer_regional(ctx, ifm, rhs, dst, attributes, option=option)
    return None if product is None else product.emit(ctx)
