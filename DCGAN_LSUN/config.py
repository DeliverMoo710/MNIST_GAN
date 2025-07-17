import os

def get_config():
    return {
        "LR" : 2e-4,
        "BATCH_SIZE" : 32,
        "LATENT_DIM" : 512,
        "IMAGE_DIM" : 64 * 64 * 3,
        "NUM_EPOCHS" : 50,
        "WEIGHT_PATH" : 'weights'
    }

# Get file path for model weights
def get_weights_file_path(config, epoch):
    return os.path.join(config["WEIGHT_PATH"], f"{epoch}.pth")