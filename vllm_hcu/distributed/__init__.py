# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU distributed implementations.

Importing the package must stay side-effect free.  Connector registration and
custom-allreduce module exchange are armed explicitly by their feature groups.
"""
