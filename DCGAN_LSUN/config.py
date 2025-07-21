import os

def get_config():
    return {
        "LR" : 2e-4,
        "BATCH_SIZE" : 32,
        "LATENT_DIM" : 512,
        "IMAGE_DIM" : 64 * 64 * 3,
        "NUM_EPOCHS" : 50,
        "WEIGHT_PATH" : 'weights',
        "name" : 'trial',
        "display_dir" : './logs',
        "result_dir" : './results',
        "display_freq" : 1,
        "img_save_freq" : 1,
        "model_save_freq" : 1,
        'resume': None
    }

# Get file path for model weights
def get_weights_file_path(config, epoch):
    return os.path.join(config["WEIGHT_PATH"], f"{epoch}.pth")