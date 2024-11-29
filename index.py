import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and display the images horizontally
img1 = cv2.imread('./imgs/img1.png')  
img2 = cv2.imread('./imgs/img2.png') 

if img1 is None or img2 is None:
    raise FileNotFoundError("Images not found. Ensure the file paths are correct.")

# Convert to RGB for matplotlib display
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# Step 2: SIFT feature extraction
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

question = input("Enter question number [1,2,3] default 1\n") or "1"


# Display images horizontally
if(question == "1"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"# features in Image 1: {len(keypoints1)}")
    plt.imshow(img1_rgb)

    plt.subplot(1, 2, 2)
    plt.title(f"# features in Image 2: {len(keypoints2)}")
    plt.imshow(img2_rgb)
    plt.show()
    

# Detect keypoints and descriptors

# Step 3: Brute Force Matcher and display matches
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

if(question == "2"):
    # Draw matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert to RGB for display
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.title("Brute Force Matches")
    plt.imshow(matched_image_rgb)
    plt.show()


# Step 4: Find homography and warp the second image over the first
if(question == "3"):
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RHO, 5.0)

    # Apply translation transformation
    trans_x = 0
    trans_y = 500
    trans = np.float32([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]])

    # Create output canvas
    out = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = cv2.warpPerspective(img1, trans, (3000, 2000), out, borderMode=cv2.BORDER_TRANSPARENT)

    # Warp the second image onto the canvas
    out = cv2.warpPerspective(img2, trans @ H, (3000, 2000), out, borderMode=cv2.BORDER_TRANSPARENT)

    # Display the final mosaic
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.title("Mosaic Output")
    plt.imshow(out_rgb)
    plt.show()
