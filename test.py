import kagglehub

# Download latest version
kagglehub.dataset_download("easterharry/vlsp-2016", "./")
amazon_dataset_path_hub = kagglehub.dataset_download("rogate16/amazon-reviews-2018-full-dataset", "./")

print("Path to dataset files:", path)