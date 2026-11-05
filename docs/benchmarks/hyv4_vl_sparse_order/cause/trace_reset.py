worker._ready_records=[]
worker._ready_trace=bool(params.get('enabled',False))
result={'rank':worker.rank,'enabled':worker._ready_trace}
