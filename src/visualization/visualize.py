import matplotlib.pyplot as plt
import os
import numpy as np



# Visualise the predicted mask in 3D using a different colour for each class
def plot_3d_mask_multiclass(mask_all,
                            mask_colors, 
                            title,
                            results_dir,
                            filename,
                            tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"]) -> None:
    """_summary_

    Args:
        mask_all (np.array): the mask with all classes present
        mask_colors (np.array): the original one hot encoded mask, each class dimension is used to specify a colour
        title (str): title of the plot
        results_path (str): directory to write the plot to
        filename (str): name of output plot file 
        tissue_labels (list, optional): Defaults to ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"].
    """
    
    # Initialie figure
    fig = plt.figure(figsize=(10, 10))
    
    # Add subplot to figure
    ax = fig.add_subplot(111, projection='3d')


    # For each class in the predicted mask, visualise the mask in 3D using a different colour
    # for i in range(mask.shape[0]):
    #     ax.voxels(mask[i,:,:,:], edgecolor='k', facecolors=)

    # Build up the colors mask using the indvidual segmentation masks

    print("Setting up colors mask")
    print(f"Mask colors shape: {mask_colors.shape}")
    colors = np.zeros_like(mask_all)
    colors[mask_colors[0,:,:,:] == 1] = "red"
    colors[mask_colors[1,:,:,:] == 1] = "blue"
    colors[mask_colors[2,:,:,:] == 1] = "green"
    colors[mask_colors[3,:,:,:] == 1] = "yellow"

    print("Number of red voxels:", np.sum(colors == "red"))
    print("Number of blue voxels:", np.sum(colors == "blue"))
    print("Number of green voxels:", np.sum(colors == "green"))
    print("Number of yellow voxels:", np.sum(colors == "yellow"))

    ax.voxels(mask_all, edgecolor='k', facecolors=colors)

    ax.set_title(title)
    plt.legend(tissue_labels)
    plt.savefig(os.path.join(results_dir, filename), bbox_inches="tight", dpi=500)

    print(f"Saved figure to {os.path.join(results_dir, filename)}")




# Loop through all the paths to the predicted segmentation masks, load the masks and visualise them in 3D
def plot_all_3d_masks_multiclass(mask_paths, 
                                results_path,
                                tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"],
                                remove_background=True) -> None:
    
    # Loop through all the predicted masks
    for i, mask_path in enumerate(mask_paths):
        
        print(f"Visualising mask {i+1}/{len(mask_paths)}")

        # Load mask using mask_path
        mask = np.load(mask_path)

        # Remove dimension of one from mask
        mask = np.squeeze(mask)
        
        # Remove background from mask
        if remove_background:
            print(f"Mask shape: {mask.shape}")
            print("Removing background mask")
            mask = mask[1:,:,:,:]
            print(f"Mask shape after removing background: {mask.shape}")

        # Convert mask so there are 1s if any class is present
        mask_all = np.any(mask, axis=0).astype(int)
        print(f"Mask all shape: {mask_all.shape}") 

        # Define the title for the plot
        title = f"{mask_path}: Predicted Segmentation Mask"
        
        # Define the filename for the plot
        filename = f"{os.path.splitext(os.path.basename(mask_path))[0]}_predicted_mask.png"
        
        # Visualise the predicted mask in 3D
        plot_3d_mask_multiclass(mask_all, mask, title, results_path, filename, tissue_labels)
    







if __name__ == "__main__":

    # Create list of paths to the predicted masks from processed data folder
    pred_masks_dir = "/mnt/scratch/scjb/data/processed/pred_masks"
    figures_dir = "/mnt/scratch/scjb/results/figures"

    mask_paths = os.listdir(pred_masks_dir)
    print("Number of predicted masks:", len(mask_paths))
    print(f"Masks to be plotted: {mask_paths}")

    # Filter for numpy files
    mask_paths = [os.path.join(pred_masks_dir, mask_path) for mask_path in mask_paths if mask_path.endswith(".npy")]
    print("Number of predicted masks after filtering:", len(mask_paths))
    print("Mask paths after filtering:", mask_paths)

    # Visualise all the predicted masks in 3D
    plot_all_3d_masks_multiclass(mask_paths, figures_dir)