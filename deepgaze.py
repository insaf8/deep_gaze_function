import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from PIL import Image, ImageDraw
import csv
import cv2
import os
import matplotlib.pyplot as plt
from deepgaze_pytorch import DeepGazeIII
from scipy.ndimage import gaussian_filter

DEVICE = 'cpu'

def predict_scanpaths(image_path, number_scanpaths):
    save_filepath = r''

    # Read the image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    
    # Resize the image
    image = image.resize((int(w/2.5), int(h/2.5)))
    w, h = image.size
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Load centerbias template
    # Expand ~ to the user's home directory
    centerbias_template_path = os.path.expanduser('~/area_interest_attention_heat_map(deepgaze)/deepgaze_pytorch/centerbias_mit1003.npy')

# Load the .npy file
    centerbias_template = np.load(centerbias_template_path)
    #centerbias_template = np.load(r'~/area_interest_attention_heat_map(deepgaze)/deepgaze_pytorch/deepgaze_pytorch/centerbias_mit1003.npy')
    
    # Rescale centerbias to match image size
    centerbias = zoom(centerbias_template, (h / centerbias_template.shape[0], w / centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)
    
    # Initialize fixations
    fixations_x = [w // 2]
    fixations_y = [h // 2]
    fixations = 1
    
    scanpaths = []
    
    # Predict scanpaths
    while len(scanpaths) < number_scanpaths:
        # Create circular mask for inhibition of return
        radius = int(0.2 * min(w, h))
        mask = create_circular_mask(h, w, fixations_x, fixations_y, radius)
        
        # Predict next fixation
        brightest_pixel = prediction(image_np * mask.unsqueeze(2).numpy().astype('uint8'), fixations_x, fixations_y, centerbias, fixations, mask)
        
        # Add predicted fixation to the list
        fixations_x.append(brightest_pixel[3])
        fixations_y.append(brightest_pixel[2])
        
        # Increment fixations parameter
        if fixations <= 3:
            fixations += 1
        
        # Append the predicted scanpath
        scanpaths.append((brightest_pixel[3], brightest_pixel[2]))
    
    # Generate attention heatmap
    heatmap = generate_attention_heatmap(w, h, scanpaths, save_filepath)
    mask = apply_heatmap_mask(image_path, heatmap, save_filepath)
    box = draw_scanpath_with_significance_on_image(image_gray, scanpaths, save_filepath)
    
    return scanpaths, heatmap, mask, box

def prediction(image, fixations_x, fixations_y, centerbias, fixations, mask):
    # Location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
    fixation_history_x = np.array(fixations_x)
    fixation_history_y = np.array(fixations_y)

    model = DeepGazeIII(fixations, pretrained=True).to(DEVICE)

    image_tensor = torch.tensor([image[:,:,:3].transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
    x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)

    log_density_prediction = (100 + model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)) \
                                * mask.to(DEVICE).unsqueeze(0).unsqueeze(0) - (1 - mask.to(DEVICE)) * 1000

    # Find the brightest pixel in the probability map
    brightest_pixel = (log_density_prediction==torch.max(log_density_prediction)).nonzero()[0].detach().cpu().numpy()

    return brightest_pixel

def create_circular_mask(h, w, fixations_x, fixations_y, radius):
    # Get the circular mask
    mask = torch.zeros(h, w)
    Y, X = np.ogrid[:h, :w]
    for i in range(len(fixations_x)):
        dist = np.sqrt((X - fixations_x[i])**2 + (Y - fixations_y[i])**2)
        mask = torch.maximum(mask, torch.from_numpy(dist <= radius) * (1 - 1/10 * (len(fixations_x) - i - 1)))

    return 1 - mask

def generate_attention_heatmap(width, height, scanpaths, save_filepath=None, blur_radius=120):
    # Create an empty canvas for the heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Iterate through each point in the scanpaths
    max_order = len(scanpaths)
    for order, (x, y) in enumerate(scanpaths):
        # Create a meshgrid of coordinates
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate distance from each pixel to the current point
        distance = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        
        # Adjust blur radius based on order
        adjusted_blur_radius = blur_radius * (max_order - order) / max_order
        
        # Create a mask to identify pixels within the adjusted blur radius
        mask = distance <= adjusted_blur_radius
        
        # Calculate intensity based on distance and order within the blur radius
        intensity = (1 - distance[mask] / adjusted_blur_radius) * (max_order - order) / max_order  # Linearly decreasing intensity
        
        # Add intensity to the heatmap
        heatmap[mask] += intensity

    # Normalize the heatmap
    heatmap /= np.max(heatmap)
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_image.save(os.path.join(save_filepath, r'heatmap.png'))
    heatmap = np.array(heatmap_image)
    return heatmap

# The rest of your functions remain unchanged...



def read_img(image_path):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    if image is None:
        raise ValueError(f"Unable to read the image from {image_path}")
        
        # Resize the image
    image = image.resize((int(w/2.5), int(h/2.5)))
    image = np.array(image)
    return image
def org_img(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    
    w, h = image.size
    
    # Resize the image
    resized_image = image.resize((int(w*2.5), int(h*2.5)))
    
    return resized_image
def save_scanpaths_to_csv(scanpaths, image_path, csv_file):

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Order', 'X', 'Y', 'Image_Path'])
        for order, (x, y) in enumerate(scanpaths):
            writer.writerow([order, x, y, image_path])


def apply_heatmap_mask(image_path, heatmap,save_filepath=None ,colormap=cv2.COLORMAP_JET, alpha=0.4):
    # Read the image
    image=read_img(image_path)

    # Apply the colormap to the heatmap

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)

    # Convert the colormap from BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create a binary mask where the heatmap intensity is non-zero
    mask = (heatmap > 0).astype(np.uint8)

    # Apply Gaussian blur to the mask
    blurred_mask = cv2.GaussianBlur(mask.astype(float), (5, 5), 4)

    # Where the mask is zero, we want the original image to show through
    heatmap_color[blurred_mask == 0] = image[blurred_mask == 0]

    # Blend the heatmap and the original image
    blended = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)
    mask_image = Image.fromarray((blended * 1).astype(np.uint8))
    mask_image.save(os.path.join(save_filepath, r'mask_test.png'))
    return blended




def draw_scanpath_with_significance_on_image(image, scanpaths, save_filepath=None):
    # Convert NumPy array to PIL image
    max_order = len(scanpaths)
    image_pil = Image.fromarray(image)

    # Convert the image to RGBA mode to allow transparency
    image_rgba = image_pil.convert('RGBA')
    draw = ImageDraw.Draw(image_rgba)
    
    for order, (x, y) in enumerate(scanpaths):
        # Calculate significance percentage based on order
        significance = (max_order - order) / max_order
        
        # Calculate box size based on significance
        box_size = int(50 * significance)  # Adjust box size based on significance
        half_box = box_size // 2
        
        # Calculate coordinates of the box
        x1 = x - half_box
        y1 = y - half_box
        x2 = x + half_box
        y2 = y + half_box
        
        # Draw rectangle with transparency
        draw.rectangle([x1, y1, x2, y2], outline='black', fill=None)
        
        # Draw text indicating significance
        draw.text((x - 50, y - 50), f'{significance:.2%}', fill='red')
    
    # Save the image if save_filepath is provided
    if save_filepath:
        image_rgba.save(os.path.join(save_filepath, r'scanpath_with_significance.png'))
    return image_rgba






import argparse

# Your existing imports and functions here...

def main(image_path, number_scanpaths):
    # Call your predict_scanpaths function with provided image path
    predicted_scanpaths, attention_heatmap, mask, box = predict_scanpaths(image_path, number_scanpaths)
    plt.imshow(attention_heatmap, cmap='hot', interpolation='nearest')
    plt.show()

    plt.imshow(mask)
    plt.show()

    plt.imshow(box)
    plt.show()
    # Rest of your code for visualization and processing...

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict scanpaths and generate visualizations.')
    
    # Add arguments
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--number_scanpaths', type=int, default=5, help='Number of scanpaths to predict (default: 5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with provided arguments
    main(args.image_path, args.number_scanpaths)
    print("Deepgaze successfully executed.")

    
