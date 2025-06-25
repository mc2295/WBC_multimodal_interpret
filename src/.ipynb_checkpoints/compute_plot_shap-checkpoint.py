
from interpretation.plot_shap import load_dataset_features,get_region_features,load_shap_values,plot_combined_shap
from interpretation.compute_shap import compute_shap


dataset_names = ['tianjin', 'barcelona', 'rabin']
X_all = {}
shap_all = {}
region_features_all = {}

for source in dataset_names:
    print(f"Processing: {source}")
    compute_shap(source)
    columns_tabular_reduced, X = load_dataset_features(source)
    region_features_all[source] = get_region_features(columns_tabular_reduced)
    X_all[source] = X
    shap_all[source] = load_shap_values(source)
    print(X_all[source].shape, shap_all[source][0].shape)

# Plot all regions
for region in [ 'nucleus', 'cyto', 'cell']:
    plot_combined_shap(region, region_features_all, X_all, shap_all, dataset_names)