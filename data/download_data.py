import os
import subprocess

def download_dataset(dataset_name, path_to_save):
    """
    Downloads a dataset from Kaggle using the Kaggle CLI.
    """
    # Ensure the save path exists
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    # Change to the directory where you want to save the dataset
    os.chdir(path_to_save)

    # Execute the Kaggle API command
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name, '--unzip'], check=True)
    print(f"Dataset downloaded and extracted to: {path_to_save}")

def main():
    dataset_name = 'nicolassilvanash/milair-dataset'  # Correct dataset path on Kaggle
    path_to_save = './data/'

    download_dataset(dataset_name, path_to_save)

if __name__ == '__main__':
    main()
