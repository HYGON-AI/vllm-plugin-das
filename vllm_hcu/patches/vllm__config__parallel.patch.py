# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ParallelConfig: lightly context parallel fields.

PATCHES = [
################ lightly cp###########################
(
'''
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """
''',
'''
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    enable_lightly_cp: bool = False
    """Use lightly context parallel."""

    enable_lightly_cplb: bool = False
    """Use lightly context parallel load balancing."""

    enable_custom_sp: bool = False
    """Use HCU custom runtime sequence parallelism."""

'''
),
################ lightly cp###########################
]
