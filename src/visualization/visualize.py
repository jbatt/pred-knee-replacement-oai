import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import argparse
import datetime
import plotly.graph_objects as go

# TODO: Add docstrings to all functions
# TODO: Include a function to plot the original image and the predicted mask side by side


# Visualise the predicted mask in 3D using a different colour for each class
def plot_3d_mask_multiclass(mask_all,
                            mask_colors, 
                            title,
                            results_dir,
                            filename,
                            tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"]) -> None:
    """Visualise the predicted mask in 3D using a different colour for each class

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

    print(f"Mask all shape: {mask_all.shape}")
    print(f"Mask colors shape: {mask_colors.shape}")

    # Reorder the mask to match ax.voxels() function so the filled[0,0,0] corresponds to the bottom left corner of the plot
    mask_all = np.transpose(mask_all, (0, 2, 1))
    # Also transpose the mask_colors to match the mask_all
    mask_colors = np.transpose(mask_colors, (0, 1, 3, 2))

    print(f"Mask all shape after transposing: {mask_all.shape}")
    print(f"Mask colors shape after transposing: {mask_colors.shape}")

    # Build up the colors mask using the indvidual segmentation masks
    print("Setting up colors mask")

    print(f"Mask colors shape: {mask_colors.shape}")

    face_colors = np.zeros(mask_all.shape + (4,), dtype=float)  # RGBA array
    edge_colors = np.zeros(mask_all.shape + (4,), dtype=float)  # RGBA array

    # Define colors as RGBA tuples
    red = [1, 0, 0, 1]     # Red
    blue = [0, 0, 1, 1]    # Blue
    green = [0, 1, 0, 1]   # Green
    yellow = [1, 1, 0, 1]  # Yellow

    # Assign colors to voxels
    face_colors[mask_colors[0,:,:,:] == 1] = red
    face_colors[mask_colors[1,:,:,:] == 1] = blue
    face_colors[mask_colors[2,:,:,:] == 1] = green
    face_colors[mask_colors[3,:,:,:] == 1] = yellow

    print("Number of red voxels:", np.sum(face_colors == red))
    print("Number of blue voxels:", np.sum(face_colors == blue))
    print("Number of green voxels:", np.sum(face_colors == green))
    print("Number of yellow voxels:", np.sum(face_colors == yellow))

    # Use the same colors for edges
    edge_colors = face_colors.copy()

    # Create voxel plot
    ax.voxels(mask_all, facecolors=face_colors, edgecolors=edge_colors)

    ax.set_title(title)

    # Create legend with all plot colors
    red_patch = mpatches.Patch(color='red', label='Femoral cartilage')
    blue_patch = mpatches.Patch(color='blue', label='Tibial cartilage')
    green_patch = mpatches.Patch(color='green', label='Patellar cartilage')
    yellow_patch = mpatches.Patch(color='yellow', label='Meniscus')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])

    # Set plot title

    plt.savefig(os.path.join(results_dir, filename), bbox_inches="tight", dpi=500)

    print(f"Saved figure to {os.path.join(results_dir, filename)}")




def plot_3d_mask_multiclass_plotly(mask, results_figures_dir, filename) -> None:

    mask = mask.squeeze()
    print(f"Segmentation mask shape: {mask.shape}")

    # Convert segmentation mask from one hot encoding to single channel with classes as different values
    segmentation_single = np.argmax(mask, axis=0)

    colorscale = [
        [0.0, 'purple'],
        [0.2, 'purple'],
        [0.2, 'green'],
        [0.4, 'green'],
        [0.4, 'blue'],
        [0.6, 'blue'],
        [0.6, 'yellow'],
        [0.8, 'yellow'],
        [0.8, 'red'],
        [1.0, 'red']
    ]

    
    # Define an opacityscale to hide the background.
    # Normalized values for each class boundary (based on isomin and isomax) are:
    # 0: ~0.1, 1: ~0.3, 2: ~0.5, 3: ~0.7, 4: ~0.9.
    # Here we set the opacity for the background region ([0.0, 0.2]) to 0.
    opacityscale = [
        [0.0, 0.0],
        [0.2, 0.0],
        [0.2, 0.8],
        [0.4, 0.8],
        [0.4, 0.8],
        [0.6, 0.8],
        [0.6, 0.8],
        [0.8, 0.8],
        [0.8, 0.8],
        [1.0, 0.8]
    ]

    # # Define a discrete color scale for 5 classes
    # discrete_colorscale = [
    #     [0.00, "rgba(0, 0, 0, 0)"],  # Background (Transparent)
    #     [0.01, "blue"],              # Class 1
    #     [0.25, "red"],               # Class 2
    #     [0.50, "green"],             # Class 3
    #     [0.75, "yellow"],            # Class 4
    #     [1.00, "purple"],            # Class 5 (optional extra)
    # ]


    # # Generate a synthetic 3D segmentation mask (Replace with your actual data)
    # segmentation = np.zeros((50, 50, 50))
    # segmentation[10:40, 10:40, 10:40] = 1  # Example: A cube in the center

    # Create 3D coordinate grid
    x, y, z = np.mgrid[:segmentation_single.shape[0], :segmentation_single.shape[1], :segmentation_single.shape[2]]

    # Convert segmentation data to 1s and 0s (binary mask)
    volume_data = segmentation_single.astype(np.uint8)

    # Create a volume rendering
    fig = go.Figure(data=go.Volume(
        x=z.flatten(),  # X coordinates
        y=y.flatten(),  # Y coordinates
        z=x.flatten(),  # Z coordinates
        value=volume_data.flatten(),  # Flattened segmentation mask values
        isomin=-0.5,
        isomax=4.5,
        opacity=1,  # Adjust opacity for better visibility
        surface_count=5,  # Number of contour surfaces
        colorscale=colorscale,  # Color mapping
        opacityscale=opacityscale,
        colorbar=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['Background', 'Femoral Cart.', 'Tibiial Cart.', 'Patellar Cart.', 'Meniscus']
        )
    ))

    # Show the figure
    fig.write_image(os.path.join(results_figures_dir, f"{filename}_plotly.png"))
    fig.write_html(os.path.join(results_figures_dir, f"{filename}_plotly.html"))

    






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
        filename = f"{os.path.splitext(os.path.basename(mask_path))[0]}_predicted_mask"
        
        # Visualise the predicted mask in 3D
        # plot_3d_mask_multiclass(mask_all, mask, title, results_path, filename, tissue_labels)
    
        plot_3d_mask_multiclass_plotly(mask, results_path, filename)






if __name__ == "__main__":

    # Argument parser to take in project name, the model name, the predicted masks dir and the output figures dir
    parser = argparse.ArgumentParser(description="Visualise the predicted masks in 3D")
    parser.add_argument("--project_name", type=str, help="Name of the project", default="oai_subset_knee_cart_seg")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--pred_masks_dir", type=str, help="Path to the predicted masks", default="/mnt/scratch/scjb/data/processed")
    parser.add_argument("--results_dir", type=str, help="Top level figures dir to save the figures", default="/mnt/scratch/scjb/results")

    args = parser.parse_args()

    # Log start date and time and use as the name of output figures directory
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create figures directory
    figures_dir = os.path.join(args.results_dir, args.project_name, "figures", args.model_name, start_time)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures directory created: {figures_dir}")

    pred_masks_dir = os.path.join(args.pred_masks_dir, args.project_name, "pred_masks", args.model_name)
    print(f"Predicted masks directory: {pred_masks_dir}")

    # Create list of paths to the predicted masks from processed data folder
    # pred_masks_dir = "/mnt/scratch/scjb/data/processed/pred_masks"
    # figures_dir = "/mnt/scratch/scjb/results/figures"

    mask_paths = os.listdir(pred_masks_dir)
    print("Number of predicted masks:", len(mask_paths))
    print(f"Masks to be plotted: {mask_paths}")

    # Filter for numpy files
    mask_paths = [os.path.join(pred_masks_dir, mask_path) for mask_path in mask_paths if mask_path.endswith(".npy")]
    print("Number of predicted masks after filtering:", len(mask_paths))
    print("Mask paths after filtering:", mask_paths)

    # Visualise all the predicted masks in 3D
    plot_all_3d_masks_multiclass(mask_paths, figures_dir, remove_background=False)