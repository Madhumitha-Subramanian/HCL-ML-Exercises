
import cv2
import os
import matplotlib.pyplot as plt

# Define image folders
input_folder = "images"
output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection
    sobel_x = cv2.Sobel(
        blurred, cv2.CV_64F, 1, 0, ksize=3
    )
    sobel_y = cv2.Sobel(
        blurred, cv2.CV_64F, 0, 1, ksize=3
    )

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel = cv2.addWeighted(
        sobel_x, 0.5, sobel_y, 0.5, 0
    )

    # Canny edge detection
    canny = cv2.Canny(blurred, 100, 200)

    # Save output images
    name = os.path.splitext(filename)[0]

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_sobel.png"),
        sobel
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_canny.png"),
        canny
    )

    # Display results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sobel, cmap="gray")
    plt.title("Sobel Edge Detection")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(canny, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Processed: {filename}")

print("Edge detection completed!")

