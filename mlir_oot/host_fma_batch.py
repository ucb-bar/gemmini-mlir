"""Legacy symbol binding for Merlin's explicitly selected host FMA batch."""

from merlin.runtime.host_arithmetic import SourceFmaBatchCapability as _SharedBatch


class SourceFmaBatchCapability(_SharedBatch):
    def header(self) -> str:
        return super().header(namespace="gemmini_host_source")
