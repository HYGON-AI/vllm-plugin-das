try:
    import vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.du_swift_connector
    import vllm_hcu.distributed.device_communicators.custom_all_reduce
except ImportError:
    pass
