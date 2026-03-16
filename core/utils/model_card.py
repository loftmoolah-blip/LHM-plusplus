# -*- coding: utf-8 -*-
# @Function: LHM++ model registry for HuggingFace and ModelScope

ModelScope_Prior_MODEL_CARD = {
    "LHMPP": "Damo_XR_Lab/LHMPP-Prior",
}
HuggingFace_Prior_MODEL_CARD = {
    "LHMPP": "3DAIGC/LHMPP-Prior",
}
ModelScope_MODEL_CARD = {
    "LHMPP-700M": "Damo_XR_Lab/LHMPP-700M",
    # "LHMPP-700MC": "Damo_XR_Lab/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "Damo_XR_Lab/LHMPPS-700M",
}

HuggingFace_MODEL_CARD = {
    "LHMPP-700M": "3DAIGC/LHMPP-700M",
    # "LHMPP-700MC": "3DAIGC/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "3DAIGC/LHMPPS-700M",
}

MODEL_CONFIG = {
    "LHMPP-700M": "./configs/train/LHMPP-any-view.yaml",
    # "LHMPP-700MC": "./configs/train/LHMPP-any-view-convhead.yaml",  # coming soon
    "LHMPPS-700M": "./configs/train/LHMPP-any-view-DPTS.yaml",
}

MEMORY_MODEL_CARD = {
    "LHMPP-700M": 8000,  # 8G
    # "LHMPP-700MC": 8000,  # 8G, coming soon
    "LHMPPS-700M": 8000,  # 8G
}