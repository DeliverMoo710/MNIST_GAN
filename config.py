import os

# get configuration settings for the model training
def get_config():
    return {
        "lr" : 3e-4,
        "latent_dim" : 3072,
        "image_dim" : 32 * 32 * 3, # 3072
        "batch_size" : 32,
        "num_epochs" : 50,
        "weight_path" : "weights/cifar10",
    }

# Get file path for model weights
def get_weights_file_path(config, type, suffix):
# config: dict
# type: str, "encoder" or "decoder"
# suffix: str, e.g. "best", "last"
    return os.path.join(config["weight_path"], type, f"{suffix}.pth")
