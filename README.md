# Image Processing App🎉
This is a Streamlit-based web application for performing various image processing operations, such as adding noise, applying blur, edge detection, Hough transforms, and morphological operations. The app features a user-friendly interface with a background image.
Feature

Image Upload: Upload images in PNG, JPG, JPEG, or GIF formats.
Image Processing Operations:
Convert images to RGB or grayscale.
Add noise (Gaussian, Salt & Pepper, Poisson).
Apply blur (Average, Gaussian, Median).
Adjust brightness and contrast, compute histograms, and equalize histograms.
Perform edge detection (Sobel, Prewitt, Roberts, Laplacian, Canny, etc.).
Detect lines and circles using Hough transforms.
Apply morphological operations (Dilation, Erosion, Open, Close).

Prerequisites

Python 3.8 or higher
A modern web browser (Chrome, Firefox, Edge, etc.)
Git (for cloning the repository)

Installation

Clone the Repository:
git clone https://github.com/Ahmed-7-ML/ImageProcessingApp.git
cd image-processing-app


Create a Virtual Environment (optional but recommended):
python -m venv venv
On Windows: venv\Scripts\activate


Install Dependencies:Ensure you have a requirements.txt file in the repository root with the following content:
streamlit
opencv-python
numpy
matplotlib

Install the dependencies:
pip install -r requirements.txt


Prepare the Background Image:

The app uses bg.jpg as the background image, which must be placed in the repository root (or update the image_path variable in streamlit_app.py to point to your image).
Supported formats: JPEG, PNG.

Usage
Run the App:
streamlit run streamlit_app.py

This will start a local server, and the app will open in your default web browser (typically at http://localhost:8501).

Interact with the App:

Upload an Image: Use the file uploader in the left column to upload an image (PNG, JPG, JPEG, or GIF).
Select a Processing Option: Choose a tab (e.g., "Add Noise", "Blur") from the "Image Processing Tools" section.
Adjust Parameters: Use dropdowns and sliders to configure processing options (e.g., noise amount, blur kernel size).
View Results: The original image appears in the left column, and the processed image (or histogram) appears in the right column within the selected tab.
Toggle Theme: Switch between Light and Dark themes using the sidebar radio button (changes apply instantly).
Reset Image: Click the "Reset Image" button in the sidebar to clear the uploaded image and start over.


Tips:

For operations like Hough Transform or edge detection, use high-quality images for better results.
If the background image (bg.jpg) is not visible, ensure it’s in the correct path and is a valid JPEG/PNG file.

File Structure
image-processing-app/
├── streamlit_app.py  # Main Streamlit application script
├── bg.jpg            # Background image for the app
├── requirements.txt  # Python dependencies
└── README.md         # This file


streamlit_app.py: Contains the core logic, including image processing functions, UI layout, and CSS styling.
bg.jpg: The default background image (replace with your own if desired).
requirements.txt: Lists required Python packages for easy installation.

Deployment to Streamlit Cloud
To deploy the app to Streamlit Cloud:

Push to GitHub:

Create a GitHub repository and push your code:git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/image-processing-app.git
git push -u origin main


Set Up Streamlit Cloud:

Sign in to Streamlit Cloud using your GitHub account.
Click "New app" and select your repository (image-processing-app).
Specify the main script path: streamlit_app.py.
Ensure requirements.txt is included in the repository root.
Click "Deploy" to launch the app.


Notes:

Ensure bg.jpg is included in the repository and accessible at the root level.
Streamlit Cloud will automatically install dependencies from requirements.txt.
If the app fails to load, check the Streamlit Cloud logs for errors (e.g., missing files or unsupported Python versions).


Troubleshooting

App Doesn’t Start:

Ensure you’re running streamlit run streamlit_app.py, not python streamlit_app.py.
Verify all dependencies are installed (pip install -r requirements.txt).
Check Python version (python --version); 3.8+ is recommended.


Background Image Not Showing:

Confirm bg.jpg exists in the repository root or update image_path in streamlit_app.py.
Ensure the file is a valid JPEG/PNG and not corrupted.


Processed Images Not Displaying:

Verify the uploaded image is in a supported format (PNG, JPG, JPEG, GIF).
Check for errors in the terminal or browser console (F12).
Test with smaller images if processing is slow.


Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a Pull Request on GitHub.

Happy image processing! 🎉
