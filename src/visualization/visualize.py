import matplotlib.pyplot as plt
import os
import numpy as np



# Visualise the predicted mask in 3D using a different colour for each class
def plot_3d_mask_multiclass(mask, 
                            title,
                            results_path,
                            filename,
                            tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"]) -> None:
    
    # Initialie figure
    fig = plt.figure(figsize=(10, 10))
    
    # Add subplot to figure
    ax = fig.add_subplot(111, projection='3d')
    
    # For each class in the predicted mask, visualise the mask in 3D using a different colour
    for i in range(mask.shape[0]):
        ax.voxels(mask[i,:,:,:], edgecolor='k', facecolors=f"C{i}")
        
    ax.set_title(title)
    plt.legend(tissue_labels)
    plt.savefig(os.path.join(results_path, filename), bbox_inches="tight", dpi=500)
    plt.show()





# Loop through all the paths to the predicted segmentation masks, load the masks and visualise them in 3D
def plot_all_3d_masks_multiclass(mask_paths, 
                                results_path,
                                tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"]) -> None:
    
    # Loop through all the predicted masks
    for i, mask_path in enumerate(mask_paths):
        
        # Load mask using mask_path
        mask = np.load(mask_path)

        # Remove dimension of one from mask
        mask = np.squeeze(mask)

        # Define the title for the plot
        title = f"{mask_path}: Predicted Segmentation Mask"
        
        # Define the filename for the plot
        filename = f"{os.path.splitext(os.path.basename(mask_path))[0]}_predicted_mask.png"
        
        # Visualise the predicted mask in 3D
        plot_3d_mask_multiclass(mask, title, results_path, filename, tissue_labels)
    







if __name__ == "__main__":

    # Create list of paths to the predicted masks from processed data folder
    pred_masks_dir = "/mnt/scratch/scjb/data/processed/pred_masks"
    figures_dir = "/mnt/scratch/scjb/results/figures"

    mask_paths = os.listdir(pred_masks_dir)

    # Filter for numpy files
    mask_paths = [os.path.join(pred_masks_dir, mask_path) for mask_path in mask_paths if mask_path.endswith(".npy")]

    # Visualise all the predicted masks in 3D
    plot_all_3d_masks_multiclass(mask_paths, figures_dir)