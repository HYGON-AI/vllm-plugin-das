<h3 align="center">
vLLM HCU Plugin
</h3>

---

## Install and build whl package

**Install**   python3 setup.py install

**build whl package** python3 setup.py bdist_wheel (If set ADD_GIT_VERSION = 1, whl will include the git number)

---

## Architecture


```
vllm_hcu
├── README.md
├── setup.py                     # Legacy build script (with C++ extensions) 
└── vllm_hcu
    ├── __init__.py
    ├── model_executor
    │   ├── __init__.py
    │   ├── layers
    │       ├── linear.py        # Custom linear
    │   └── models               # Custom models
    │   └── parameter.py         # Custom parameter
    ├── ops                      # Custom operators (rotary_embedding)
    │   ├── __init__.py
    │   ├── rotary_embedding.py
    ├── patches                  # hcu patches 
    │   ├── __init__.py
    ├── patch_utils.py
    ├── platforms                 # hcu platform design
    │   ├── envs.py               # vllm hcu envs
    │   ├── hcu.py               
    │   └── __init__.py
    ├── v1
    │   ├── attention             # hcu backend attention     
    │   ├── __init__.py
    │   ├── backends              # hcu attention backends
    │      ├── hcu_flash_attn     # hcu flash_attn    
    │      ├── hcu_triton_attn    # hcu triton_attn 
    │   └── worker.py             # hcu work   
    │   └── hcu_model_runner.py   # hcu model runner  
    └── version.py
```

---
