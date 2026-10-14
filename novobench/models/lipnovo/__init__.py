__all__ = [
    "LipNovoRunner",
]


def __getattr__(name):
    if name == "LipNovoRunner":
        from .lipnovo_runner import LipNovoRunner

        return LipNovoRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
