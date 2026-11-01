"""Optional SlimQuant packed W4 / dynamic INT8 provider (QNT009)."""


def install():
    from .lifecycle import install as activate

    return activate()
