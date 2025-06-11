import kagglehub
import os


# Download VLSP 2016 dataset
print("Downloading VLSP 2016 dataset...")
vlsp_dataset_path = kagglehub.dataset_download("easterharry/vlsp-2016")

# Download Amazon Reviews dataset  
print("Downloading Amazon Reviews dataset...")
amazon_dataset_path = kagglehub.dataset_download("rogate16/amazon-reviews-2018-full-dataset")

print("VLSP dataset path:", vlsp_dataset_path)
print("Amazon dataset path:", amazon_dataset_path)

# List files in downloaded directories
print("\nVLSP dataset files:")
for root, dirs, files in os.walk(vlsp_dataset_path):
    for file in files:
        print(os.path.join(root, file))

print("\nAmazon dataset files:")
for root, dirs, files in os.walk(amazon_dataset_path):
    for file in files[:5]:  # Chỉ hiển thị 5 file đầu tiên
        print(os.path.join(root, file))