import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import argparse
import datetime
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import scienceplots
import scipy.stats as stats

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

    # TODO: change plot background from grey
    # TODO: Add title, correct colorbar and labels for each class
    # TODO: Add comparison to the ground truth mask

    print(f"Mask shape: {mask.shape}")
    mask = mask.squeeze()
    print(f"Mask shape after squeezing: {mask.shape}")

    # Convert segmentation mask from one hot encoding to single channel with classes as different values
    segmentation_single = np.argmax(mask, axis=0)

    # Print the unique values in the segmentation mask
    print(f"Unique values in the segmentation mask: {np.unique(segmentation_single)}")

    # TODO: fix color - patellar cartilage and tibial cartilage ar both blue

    colorscale = [
        [0.0, 'purple'],
        [0.19, 'purple'],
        [0.21, 'green'],
        [0.39, 'green'],
        [0.41, 'blue'],
        [0.59, 'blue'],
        [0.61, 'yellow'],
        [0.79, 'yellow'],
        [0.81, 'red'],
        [1.0, 'red']
    ]
    
    # Define an opacityscale to hide the background.
    # Normalized values for each class boundary (based on isomin and isomax) are:
    # 0: ~0.1, 1: ~0.3, 2: ~0.5, 3: ~0.7, 4: ~0.9.
    # Set the opacity for the background region ([0.0, 0.2]) to 0.
    opacityscale = [
        [0.0, 0.0],
        [0.2, 0.0],
        [0.2, 1.0],
        [0.4, 1.0],
        [0.4, 1.0],
        [0.6, 1.0],
        [0.6, 1.0],
        [0.8, 1.0],
        [0.8, 1.0],
        [1.0, 1.0]
    ]

    # # Generate a synthetic 3D segmentation mask (Replace with your actual data)
    # segmentation = np.zeros((50, 50, 50))
    # segmentation[10:40, 10:40, 10:40] = 1  # Example: A cube in the center

    # Create 3D coordinate grid
    x, y, z = np.mgrid[:segmentation_single.shape[0], :segmentation_single.shape[1], :segmentation_single.shape[2]]

    # Convert segmentation data int8 type to save memory
    volume_data = segmentation_single.astype(np.uint8)

    # Create a volume rendering
    fig = go.Figure(data=go.Volume(
        x=z.flatten(),  # X coordinates
        y=y.flatten(),  # Y coordinates
        z=x.flatten(),  # Z coordinates
        value=volume_data.flatten(),  # Flattened segmentation mask values
        isomin=-0.5, 
        isomax=4.5, # -0.5-0.5, 0.5-1.5, 1.5-2.5, 2.5-3.5, 3.5-4.5
        # opacity=1,  # Adjust opacity for better visibility
        surface_count=5,  # Number of contour surfaces
        colorscale=colorscale,  # Color mapping
        opacityscale=opacityscale,
        colorbar=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['Background', 'Femoral Cart.', 'Tibiial Cart.', 'Patellar Cart.', 'Meniscus']
        )
    ))

    # Reverse the z-axis in the plot
    fig.update_layout(
        scene=dict(
            zaxis=dict(autorange='reversed'),
            yaxis=dict(autorange='reversed'),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Predicted 3D Segmentation Map'
)

    # Show the figure
    fig.write_image(os.path.join(results_figures_dir, f"{filename}_plotly.png"))
    fig.write_html(os.path.join(results_figures_dir, f"{filename}_plotly.html"))

    print(f"Saved figure to {os.path.join(results_figures_dir, filename)}")

    




# Loop through all the paths to the predicted segmentation masks, load the masks and visualise them in 3D
def plot_all_3d_masks_multiclass(mask_paths, 
                                results_path,
                                tissue_labels = ["Femoral cart.", "Tibial cart.", "Patellar cart.", "Meniscus"],
                                remove_background=True) -> None:
    
    # Loop through all the predicted masks
    for i, mask_path in enumerate(mask_paths):
        
        print(f"Visualising mask {i+1}/{len(mask_paths)}")
        print(f"Mask path: {mask_path}")

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
        


        # DECIDE ON INDIVIDUAL PLOTTING FUNCTIONS HERE

        # Visualise the predicted mask in 3D
        # plot_3d_mask_multiclass(mask_all, mask, title, results_path, filename, tissue_labels)
    
        plot_3d_mask_multiclass_plotly(mask, results_path, filename)



# Bland-Altman plots

# TODO: output PDF version?
# TODO: increase ick label size


def plot_bland_altman(model, run_start_time, results_dir="/mnt/scratch/scjb/results/oai_subset_knee_cart_seg"):
    
    eval_metrics_dir = os.path.join(results_dir, "eval_metrics", model, run_start_time)

    df_te = pd.read_csv(os.path.join(eval_metrics_dir, f"te_{model}_{run_start_time}.csv"))
    df_tm = pd.read_csv(os.path.join(eval_metrics_dir, f"tm_{model}_{run_start_time}.csv"))

    print(df_te.head())

    # Calculate mean thickness and thick error for each cartilage type
    te_means = df_te[["fem cart.", "tibial cart.", "patellar cart."]].mean()
    tm_means = df_tm[["fem cart.", "tibial cart.", "patellar cart."]].mean()
    te_std = df_te[["fem cart.", "tibial cart.", "patellar cart."]].std()
    tm_std = df_tm[["fem cart.", "tibial cart.", "patellar cart."]].std()


    # plt.rcParams['text.usetex'] = True # TeX rendering

    with plt.style.context(['science', 'no-latex']):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,5), sharey=True)

        axs[0].scatter(df_tm["fem cart."], df_te["fem cart."], color="black", zorder=2)
        axs[0].grid(linestyle='dotted')
        # axs[0].set_xlabel(r"$\mathrm{Thickness \ Mean}$", fontsize=16) 
        axs[0].set_xlabel("Thickness Mean (mm)", fontsize=14) 
        axs[0].set_ylabel("Thickness Error (mm)", fontsize=14) 
        axs[0].set_title("Femoral Cartilage Cartilage", fontsize=16)
        axs[0].axhline(te_means[0], xmin=0, xmax=1, color="grey", linestyle="dotted", lw=1.8, zorder=1)
        axs[0].axhline(te_means[0] + te_std[0], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)
        axs[0].axhline(te_means[0] - te_std[0], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)

        axs[1].scatter(df_tm["tibial cart."], df_te["tibial cart."], color="black", zorder=2)
        axs[1].grid(linestyle='dotted')
        axs[1].set_xlabel("Thickness Mean (mm)", fontsize=14) 
        axs[1].set_ylabel("Thickness Error (mm)", fontsize=14)
        axs[1].set_title("Tibial Cartilage", fontsize=16)
        axs[1].axhline(te_means[1], xmin=0, xmax=1, color="grey", linestyle="dotted", lw=1.8, zorder=1)
        axs[1].axhline(te_means[1] + te_std[1], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)
        axs[1].axhline(te_means[1] - te_std[1], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)

        # axs[1].set_xlabel(r"$\mathrm{Thickness \  Mean$}", fontsize=16) 

        axs[2].scatter(df_tm["patellar cart."], df_te["patellar cart."], color="black", zorder=2)
        axs[2].grid(linestyle='dotted')
        axs[2].set_xlabel("Thickness Mean (mm)", fontsize=14) 
        axs[2].set_ylabel("Thickness Error (mm)", fontsize=14) 
        axs[2].set_title("Patellar Cartilage", fontsize=16)
        axs[2].axhline(te_means[2], xmin=0, xmax=1, color="grey", linestyle="dotted", lw=1.8, zorder=1)
        axs[2].axhline(te_means[2] + te_std[2], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)
        axs[2].axhline(te_means[2] - te_std[2], xmin=0, xmax=1, color="grey", linestyle="--", lw=1.8, zorder=1)
        # axs[2].set_xlabel(r"$\mathrm{Thickness \: Mean}$}", fontsize=16) 

        plt.tight_layout()
        
        output_figure_dir = Path(os.path.join(results_dir, "figures", model, run_start_time)) # Make results directory

        output_figure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_figure_dir, f"bland_altman_{model}_{run_start_time}"), bbox_inches="tight", dpi=500)



def plot_seg_metric_thickness_error_corr(model, run_start_time, results_dir="/mnt/scratch/scjb/results/oai_subset_knee_cart_seg"):
        
    """
    Plot thickness error for 3 cartilage types against segmentation metrics 

    model: name of model
    run_start_time: the start time of the model training or inference run
    results_dir: the top-level results directory

    """
    # Create a figure with 4 columns and 3 rows

    # TODO: save figure to results dir using model name and run start time 
    # TODO: use model name and run start time to load eval metrics
    # TODO: include other pose measures

    # Load eval_metrics

    eval_metrics_dir = os.path.join(results_dir, "eval_metrics", model, run_start_time)

    df_te = pd.read_csv(os.path.join(eval_metrics_dir, f"te_{model}_{run_start_time}.csv"))
    df_dice = pd.read_csv(os.path.join(eval_metrics_dir, f"dice_{model}_{run_start_time}.csv"))
    df_hd = pd.read_csv(os.path.join(eval_metrics_dir, f"hd_{model}_{run_start_time}.csv"))
    df_assd = pd.read_csv(os.path.join(eval_metrics_dir, f"assd_{model}_{run_start_time}.csv"))

    nrows = 3
    ncols = 3

    with plt.style.context(['science', 'no-latex']):
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,16), sharey=False)

        for i in range(nrows):

            # Dice
            print("\n", df_te.columns[i+1])
            te_dice_cor = stats.pearsonr(df_dice.iloc[:,i+1], df_te.iloc[:,i+1])
            dice_linreg = stats.linregress(df_dice.iloc[:,i+1], df_te.iloc[:,i+1])
            
            # min_x = df_dice.iloc[:,i+1].min()
            # linreg_min_y = min_x*dice_linreg.slope + dice_linreg.intercept

            axs[i][0].scatter(df_dice.iloc[:,i+1], df_te.iloc[:,i+1], color="black", zorder=2)
            axs[i][0].axline(xy1=(0, dice_linreg.intercept), slope=dice_linreg.slope, color="grey", zorder=1)        
            axs[i][0].grid(linestyle='dotted')
            axs[i][0].set_title("Dice Score", fontsize=18)
            axs[i][0].tick_params(axis='x', labelsize=13)
            axs[i][0].tick_params(axis='y', labelsize=13)
            axs[i][0].annotate(f"r = {te_dice_cor.statistic:.2f}", xy=(0,1), xytext=(0.15, 0.9), xycoords='axes fraction', fontsize=16)
            axs[i][0].set_ylim(bottom=0)
            

            # Hausdorff Distance
            te_hd_cor = stats.pearsonr(df_hd.iloc[:,i+1], df_te.iloc[:,i+1])
            hd_linreg = stats.linregress(df_hd.iloc[:,i+1], df_te.iloc[:,i+1])
            print(f"HD corr: {te_hd_cor}")
            axs[i][1].scatter(df_hd.iloc[:,i+1], df_te.iloc[:,i+1], color="black", zorder=2)
            axs[i][1].axline(xy1=(0, hd_linreg.intercept), slope=hd_linreg.slope, color="grey", zorder=1)
            axs[i][1].grid(linestyle='dotted')
            axs[i][1].set_title("Hausdorff Distance", fontsize=18)
            axs[i][1].tick_params(axis='x', labelsize=13)
            axs[i][1].tick_params(axis='y', labelsize=13)
            axs[i][1].annotate(f"r = {te_hd_cor.statistic:.2f}", xy=(0,1), xytext=(0.15, 0.9), xycoords='axes fraction', fontsize=16)
            axs[i][1].set_ylim(bottom=0)

            # ASSD
            te_assd_cor = stats.pearsonr(df_assd.iloc[:,i+1], df_te.iloc[:,i+1])
            assd_linreg = stats.linregress(df_assd.iloc[:,i+1], df_te.iloc[:,i+1])
            print(f"ASSD corr: {te_assd_cor}")
            axs[i][2].scatter(df_assd.iloc[:,i+1], df_te.iloc[:,i+1], color="black", zorder=2)
            axs[i][2].axline(xy1=(0, assd_linreg.intercept), slope=assd_linreg.slope, color="grey", zorder=1)   
            axs[i][2].grid(linestyle='dotted')
            axs[i][2].set_title("Average Symmetric Surface Distance", fontsize=18)
            axs[i][2].tick_params(axis='x', labelsize=13)
            axs[i][2].tick_params(axis='y', labelsize=13)
            axs[i][2].annotate(f"r = {te_assd_cor.statistic:.2f}", xy=(0,1), xytext=(0.15, 0.9), xycoords='axes fraction', fontsize=16)
            axs[i][2].set_ylim(bottom=0)


            # Global plot formatting
            # Set y yable of each cartilage type
            axs[0][0].set_ylabel("Femoral Cart. Thickness Error", fontsize=16)
            axs[1][0].set_ylabel("Tibial Cart. Thickness Error", fontsize=16)
            axs[2][0].set_ylabel("Patellar Cart. Thickness Error", fontsize=16)

            
        plt.tight_layout()
        
        # Create figures directory
        output_figure_dir = Path(os.path.join(results_dir, "figures", model, run_start_time)) 

        output_figure_dir.mkdir(parents=True, exist_ok=True)

        # Write out figures - png and pdf versions
        plt.savefig(os.path.join(output_figure_dir, f"seg_metric_te_corr_{model}_{run_start_time}"), bbox_inches="tight", dpi=500)
        plt.savefig(os.path.join(output_figure_dir, f"seg_metric_te_corr_{model}_{run_start_time}"), bbox_inches="tight", format="pdf")



if __name__ == "__main__":

    # Argument parser to take in project name, the model name, the predicted masks dir and the output figures dir
    parser = argparse.ArgumentParser(description="Visualise the predicted masks in 3D")
    parser.add_argument("--project_name", type=str, help="Name of the project", default="oai_subset_knee_cart_seg")
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--run_start_time", type=str, help="start time of model inference run")
    parser.add_argument("--pred_masks_dir", type=str, help="Path to the predicted masks", default="/mnt/scratch/scjb/data/processed")
    parser.add_argument("--results_dir", type=str, help="Top level figures dir to save the figures", default="/mnt/scratch/scjb/results")

    args = parser.parse_args()

    print(f"Passed arg: {args}\n")

    # Log start date and time and use as the name of output figures directory
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create figures directory
    figures_dir = os.path.join(args.results_dir, args.project_name, "figures", args.model, start_time)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures directory created: {figures_dir}")
    
    pred_masks_dir = os.path.join(args.pred_masks_dir, args.project_name, "pred_masks", args.model)
    
    # nnunet outputs multiple sets of results so specify the postprocesing version of the results
    if args.model == "nnunet":
        pred_masks_dir = os.path.join(pred_masks_dir, "postprocesing")
                                      
    print(f"Predicted masks directory: {pred_masks_dir}")

    # Create list of paths to the predicted masks from processed data folder
    # pred_masks_dir = "/mnt/scratch/scjb/data/processed/pred_masks"
    # figures_dir = "/mnt/scratch/scjb/results/figures"

    mask_paths = os.listdir(pred_masks_dir)
    print("Number of predicted masks:", len(mask_paths))
    print(f"Masks to be plotted: {mask_paths}")

    # Filter for numpy files
    # mask_paths = [os.path.join(pred_masks_dir, mask_path) for mask_path in mask_paths if mask_path.endswith(".npy")]
    # print("Number of predicted masks after filtering:", len(mask_paths))
    # print("Mask paths after filtering:", mask_paths)

    # Visualise all the predicted masks in 3D
    # plot_all_3d_masks_multiclass(mask_paths, figures_dir, remove_background=False)

    # Bland Altman plots
    plot_bland_altman(args.model, args.run_start_time, results_dir="../results/")

    # Seg metric / thickness error correlation plots
    plot_seg_metric_thickness_error_corr(args.model, args.run_start_time, results_dir="../results/")