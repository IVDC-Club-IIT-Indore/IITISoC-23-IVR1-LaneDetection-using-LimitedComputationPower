from PIL import Image
import numpy as np
import glob

# Specify the directory containing the images
image_dir = r"C:\Users\ampad\Downloads\dataset\labels_lane/"   # Replace with the directory path containing your images

# Specify the desired image size
new_size = (80, 160)  # Replace with the desired size of the images

# Create an empty list to store the arrays
image_arrays = []

# Get a list of image file paths in the directory
image_paths = glob.glob(image_dir + '*.png')  # Replace '*.jpg' with the appropriate file extension

# Loop through each image, resize it, and convert it to a NumPy array
for image_path in image_paths:
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize(new_size)

    # Convert the resized image to a NumPy array
    image_array = np.array(resized_image)

    # Append the array to the list
    image_arrays.append(image_array)

# Stack the arrays along a new axis to create a single array
combined_array = np.stack(image_arrays)

# Print the shape of the combined array
print("Shape of the combined array:", combined_array.shape)

# Save the combined array as a .npy file
output_file = r'C:\Users\ampad\Downloads\dataset\lane.npy'  #Replace with the desired output file path
np.save(output_file, combined_array)
