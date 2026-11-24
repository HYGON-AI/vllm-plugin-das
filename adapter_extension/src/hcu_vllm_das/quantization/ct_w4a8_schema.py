# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared packed channel-INT4/token-INT8 checkpoint contract."""


def schema_matches(weight, activation):
    def value(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return (weight is not None and activation is not None
        and value(weight,'num_bits') == 4 and value(weight,'type') == 'int'
        and value(weight,'strategy') == 'channel' and value(weight,'symmetric',True) is True
        and value(weight,'dynamic',False) is False and value(weight,'group_size') in (None,-1)
        and value(weight,'block_structure') is None and value(activation,'block_structure') is None
        and value(activation,'num_bits') == 8 and value(activation,'type') == 'int'
        and value(activation,'strategy') == 'token' and value(activation,'symmetric',True) is True
        and value(activation,'dynamic') is True)
