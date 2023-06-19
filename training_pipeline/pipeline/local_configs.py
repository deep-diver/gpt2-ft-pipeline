DATA_PATH = "local-data/"

HYPER_PARAMETERS = {
    "epochs": {
        "type": "choice",
        "values": [1]
    },

    "optimizer_type": {
        "type": "choice",
        "values": ["Adam"],
    },

    "learning_rate": {
        "type": "float",
        "min_value": 0.00001,
        "max_value": 0.00001,
        "sampling": "log",
        "step": 10
    },

    "weight_decay": {
        "type": "choice",
        "values": [0.1]
    }
}