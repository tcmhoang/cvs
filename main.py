import torch
import wandb
import dataset
import config

def main():

    wandb.init(
        project="Retrieval-DINOv2",
        config={
            "architecture": "DINOv2 + GeM + MLP",
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SZ,
            "learning_rate": config.LEARNING_RATE,
            "triplet_margin": config.TRIPLET_MARGIN
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    failed_populateds = dataset.io_prepare(config.CACHE_DIR, {'lzupsd': None , 'vsd' : None})
    
    if len(failed_populateds) != 0:
        print('Failed to prepare', ' '.join(failed_populateds))




    
   
if __name__ == "__main__":
    main()
