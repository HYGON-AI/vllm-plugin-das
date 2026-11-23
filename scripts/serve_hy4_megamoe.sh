#!/usr/bin/env bash
# Pass the usual "vllm serve" arguments. Merge worker streams so repeated
# native runtime messages can be recognized across EP ranks.
set -eu
command -v perl >/dev/null
exec vllm serve "$@" > >(exec perl -ne '
    BEGIN { $| = 1 }
    if (/^(?:\(Worker[^()]*\) )?Warning: \(!g_use_interrupt_wait \|\| isIPC\(\)\) && "Use of non-host signal in host signal wait API\." in virtual hsa_signal_value_t rocr::core::BusyWaitSignal::WaitRelaxed\([^()]*\), [^\r\n]*\/default_signal\.cpp:\d+\r?$/) {
        next if $seen_wait++;
    }
    if (/^(?:\(Worker[^()]*\) )?\[rank \d+\] (SymmBuffer (?:ipc|fabric): [^\r\n]+)\r?$/) {
        next if $seen_buffer{$1}++;
    }
    print;
') 2>&1
