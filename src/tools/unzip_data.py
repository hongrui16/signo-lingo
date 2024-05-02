import zipfile

# Specify the path to your zip file
zip_path = 'path_to_your_zip_file.zip'
# Specify the directory where you want to extract the files
extract_to_directory = 'path_to_extract_directory'

# Create a ZipFile object
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extract_to_directory)

print("Files extracted successfully!")
