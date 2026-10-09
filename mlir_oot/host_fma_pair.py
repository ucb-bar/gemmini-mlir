"""Legacy symbol binding for Merlin's explicitly selected host FMA pair."""

from merlin.runtime.host_arithmetic import SourceFmaPairCapability as _SharedPair


class SourceFmaPairCapability(_SharedPair):
    def header(self) -> str:
        return super().header(namespace="gemmini_host_source")
