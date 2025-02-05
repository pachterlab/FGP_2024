# Load necessary libraries
library(ggplot2)
library(RColorBrewer)
library(zellkonverter)
library(SingleCellExperiment)
library(monocle3)
library(dplyr)

set.seed(2024)
setwd("~/workspace/Trajectory/FGP_2024")

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


# dyngen ------------------------------------------------------------------

# Load the .h5ad file into a SingleCellExperiment object
sce <- zellkonverter::readH5AD("data/dyngen_bif_1_10000.h5ad")

# Create the Monocle3 CellDataSet
expression_matrix <- as.matrix(assay(sce))  # Extract expression data (transposed for Monocle)
cell_metadata <- as.data.frame(colData(sce))   # Cell metadata from AnnData
gene_metadata <- as.data.frame(rowData(sce))   # Gene metadata from AnnData

cds <- new_cell_data_set(expression_data = expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_metadata)

# Preprocess the data (normalization, feature selection)
cds <- preprocess_cds(cds, num_dim = 50)

# Reduce dimensionality using UMAP
cds <- reduce_dimension(cds, reduction_method = "UMAP")

# Cluster cells
cds <- cluster_cells(cds)

# Learn the trajectory graph
cds <- learn_graph(cds)

# Visualize cells colored by pseudotime
plot_cells(cds, color_cells_by = "sim_time")

# Order cells along the trajectory (pseudotime)
cds <- order_cells(cds)

# Visualize UMAP plot with clusters
plot_cells(cds, color_cells_by = "cluster")

# Visualize cells colored by pseudotime
plot_cells(cds, color_cells_by = "pseudotime")

# Extract pseudotime and sim_time
pseudotime <- as.numeric(pseudotime(cds))
sim_time <- sce$sim_time  # Equivalent to adata.obs['sim_time']

# Remove Inf and NA values from pseudotime and sim_time
valid_indices <- is.finite(pseudotime) & is.finite(sim_time)

# Filter pseudotime and sim_time to exclude Inf or NA values
pseudotime_filtered <- pseudotime[valid_indices]
sim_time_filtered <- sim_time[valid_indices]

pseudotime_norm <- min_max_norm(pseudotime_filtered)
sim_time_norm <- min_max_norm(sim_time_filtered)

# Calculate RMSE (Root Mean Square Error)
rmse <- sqrt(mean((sim_time_norm - pseudotime_norm)^2))

# Create a scatter plot with ggplot2
ggplot() +
  geom_point(aes(x = sim_time_norm, y = pseudotime_norm), color = '#457b9d', alpha = 0.1, size = 2) +
  labs(
    x = "True time",
    y = "Monocle 3 pseudotime"
  ) +
  theme_minimal() + theme(
    axis.title.x = element_text(size = 24),  # Make x-axis label larger
    axis.title.y = element_text(size = 24),  # Adjust y-axis label size if needed
    axis.text = element_text(size = 24), 
    panel.grid.major = element_blank(),      # Remove major grid lines inside the plot
    panel.grid.minor = element_blank(),      # Remove minor grid lines inside the plot
    panel.border = element_rect(color = "black", fill = NA, linewidth=0.5 ),  # Add a frame around the plot
    legend.position = "none"                 # Remove legend if not needed
  ) +
  annotate("text", 
           label = paste("RMSE =", round(rmse, 2)), 
           x = 0.99, y = 0.05, hjust = 1, vjust = 0, size = 8, color = "black")  


# Chronocell ------------------------------------------------------------------

# Load the .h5ad file into a SingleCellExperiment object
sce <- zellkonverter::readH5AD("data/sim_demo.h5ad")

# Create the Monocle3 CellDataSet
expression_matrix <- as.matrix(assay(sce))  # Extract expression data (transposed for Monocle)
cell_metadata <- as.data.frame(colData(sce))   # Cell metadata from AnnData
gene_metadata <- as.data.frame(rowData(sce))   # Gene metadata from AnnData

cds <- new_cell_data_set(expression_data = expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_metadata)

# Preprocess the data (normalization, feature selection)
cds <- preprocess_cds(cds, num_dim = 50)

# Reduce dimensionality using UMAP
cds <- reduce_dimension(cds, reduction_method = "UMAP")

# Cluster cells
cds <- cluster_cells(cds)

# Learn the trajectory graph
cds <- learn_graph(cds)

# Visualize cells colored by sim time
plot_cells(cds, color_cells_by = "time")

# Order cells along the trajectory (pseudotime)
cds <- order_cells(cds, root_cells = 1)

# Visualize UMAP plot with clusters
plot_cells(cds, color_cells_by = "cluster")

# Visualize cells colored by pseudotime
plot_cells(cds, color_cells_by = "pseudotime")

# Extract pseudotime and sim_time
pseudotime <- as.numeric(pseudotime(cds))
sim_time <- sce$time  # Equivalent to adata.obs['sim_time']

# Remove Inf and NA values from pseudotime and sim_time
valid_indices <- is.finite(pseudotime) & is.finite(sim_time)

# Filter pseudotime and sim_time to exclude Inf or NA values
pseudotime_filtered <- pseudotime[valid_indices]
sim_time_filtered <- sim_time[valid_indices]

pseudotime_norm <- min_max_norm(pseudotime_filtered)
sim_time_norm <- min_max_norm(sim_time_filtered)

# Calculate RMSE (Root Mean Square Error)
rmse <- sqrt(mean((sim_time_norm - pseudotime_norm)^2))

# Create a scatter plot with ggplot2
ggplot() +
  geom_point(aes(x = sim_time_norm, y = pseudotime_norm), color = '#457b9d', alpha = 0.1, size = 2) +
  labs(
    x = "True time",
    y = "Monocle 3 pseudotime"
  ) +
  theme_minimal() + theme(
    axis.title.x = element_text(size = 24),  # Make x-axis label larger
    axis.title.y = element_text(size = 24),  # Adjust y-axis label size if needed
    axis.text = element_text(size = 24), 
    panel.grid.major = element_blank(),      # Remove major grid lines inside the plot
    panel.grid.minor = element_blank(),      # Remove minor grid lines inside the plot
    panel.border = element_rect(color = "black", fill = NA, linewidth=0.5 ),  # Add a frame around the plot
    legend.position = "none"                 # Remove legend if not needed
  ) +
  annotate("text", 
           label = paste("RMSE =", round(rmse, 2)), 
           x = 0.99, y = 0.05, hjust = 1, vjust = 0, size = 8, color = "black")  


# Neuron ------------------------------------------------------------------
# Load the .h5ad file into a SingleCellExperiment object
sce <- zellkonverter::readH5AD("data/neuron.h5ad")

# Create the Monocle3 CellDataSet
expression_matrix <- as.matrix(assay(sce))  # Extract expression data (transposed for Monocle)
cell_metadata <- as.data.frame(colData(sce))   # Cell metadata from AnnData
gene_metadata <- as.data.frame(rowData(sce))   # Gene metadata from AnnData

cds <- new_cell_data_set(expression_data = expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_metadata)

# Preprocess the data (normalization, feature selection)
cds <- preprocess_cds(cds, num_dim = 50)

# Reduce dimensionality using UMAP
cds <- reduce_dimension(cds, reduction_method = "UMAP")

# Cluster cells
cds <- cluster_cells(cds)

# Learn the trajectory graph
cds <- learn_graph(cds)

# Visualize cells colored by sim time
plot_cells(cds, color_cells_by = "time") 

# Order cells along the trajectory (pseudotime)
cds <- order_cells(cds)

# Visualize cells colored by pseudotime
plot_cells(cds, color_cells_by = "pseudotime")

# Extract pseudotime and sim_time
t <- pseudotime(cds)

# Convert pseudotime values to a data frame for easier saving
pseudotime_df <- data.frame(Cell = names(t), Pseudotime = t)

# Save to CSV
write.csv(pseudotime_df, "results/Neuron_Monocle3_pseudotime.csv", row.names = FALSE)
E