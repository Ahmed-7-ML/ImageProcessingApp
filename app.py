# Image Processing App[OOP Paradigm]
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
import time
import os

class ImageProcessor:
    def __init__(self):
        """Initialize Image Processor Class with an Input Image"""
        self.image = None
        self.image_rgb = None
        # Create output directory if it doesn't exist
        self.output_dir = "output_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Function to encode local image to base64
        def get_base64_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()

        # Path to your local image
        image_path = "bg.jpg"
        encoded_image = get_base64_image(image_path)

        # CSS for background image
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stTabs [data-baseweb="tab-panel"] {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
 
        st.title("Image Processing App 🎉")
        self.file = st.file_uploader("Upload an Image", ["png", "jpg", "jpeg", "gif"])
        self.load_image()
        
    def load_image(self):
        """Accept an Input Image of Type png, jpg, jpeg or gif"""
        if self.file is not None:
            # Convert File(Image) into a Numpy Array of Pixels
            img_arr = np.asarray(bytearray(self.file.read()), dtype=np.uint8)
            self.image = cv2.imdecode(img_arr, 1)  # BGR Image
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            st.image(self.image, caption="BGR Image", use_container_width=True)
            self.setup_tabs()
        else:
            st.info("Please Upload an Image to Get Started")

    def save_image(self, image, operation):
        """Save the image to the output directory using OpenCV"""
        # Generate a unique filename
        filename = f"{operation.lower()}.png"
        filepath = os.path.join(self.output_dir, filename)
        # Handle grayscale (2D) or color (3D) images
        cv2.imwrite(filepath, image)
        return filepath

    def add_salt_pepper_noise(self, image, amount=0.01, salt_vs_pepper=0.5):
        """Add Salt and Pepper Noise To Image"""
        noisy = np.copy(image)
        total_pixels = image.size // 3
        # Salt
        num_salt = np.ceil(amount * total_pixels * salt_vs_pepper).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        return noisy

    def add_poisson_noise(self, image):
        """Add Poisson Noise To Image"""
        noisy = np.random.poisson(image.astype(float)) # # Draw samples from a Poisson distribution.
        noisy = np.clip(noisy, 0, 255) # Clip (limit) the values in an array.
        return noisy.astype(np.uint8)

    def add_gaussian_noise(self, image, mean=0, var=0.01):
        """Add Gaussian Noise To Image"""
        h, w, c = image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (h, w, c))
        noisy = image / 255.0 + gauss
        noisy = np.clip(noisy, 0, 1)
        return (noisy * 255).astype(np.uint8)

    def convert_image(self, converter):
        """Convert Image into RGB/GrayScaled"""
        if converter == "RGB":
            return self.image_rgb
        elif converter == "Gray Scale":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur_image(self, blur_type, k):
        """Blurring the image"""
        if blur_type == "Average Blur":
            return cv2.blur(self.image_rgb, (k, k))
        elif blur_type == "Gaussian Blur":
            return cv2.GaussianBlur(self.image_rgb, (k, k), 0)
        elif blur_type == "Median Blur":
            return cv2.medianBlur(self.image_rgb, k)

    def adjust_brightness_contrast(self, alpha, beta):
        """Adjust Contrast and Brightness"""
        return cv2.convertScaleAbs(self.image_rgb, alpha=alpha, beta=beta)

    def compute_histogram(self):
        """Display Histogram of Image"""
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        fig, ax = plt.subplots()
        ax.plot(hist, color='black')
        ax.set_title("Grayscale Histogram")
        return fig

    def equalize_histogram(self):
        """Equalize Histogram"""
        return cv2.equalizeHist(self.gray)

    def apply_local_filter(self, filter_type, gray, k=5, t1=50, t2=150):
        """High/Low Pass Filters or Edge Detection by Sobel, Prewitt,...etc."""
        if filter_type == "Low Pass Filter (Blur)":
            return cv2.GaussianBlur(gray, (k, k), 0)
        elif filter_type == "High Pass Filter (Sharpen)":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(gray, -1, kernel)
        elif filter_type == "Edge - Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            output = cv2.magnitude(sobelx, sobely)
            return np.uint8(np.clip(output, 0, 255))
        elif filter_type == "Edge - Prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            img_prewittx = cv2.filter2D(gray, -1, kernelx)
            img_prewitty = cv2.filter2D(gray, -1, kernely)
            return cv2.add(img_prewittx, img_prewitty)
        elif filter_type == "Edge - Roberts":
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            img_robertsx = cv2.filter2D(gray, -1, kernelx)
            img_robertsy = cv2.filter2D(gray, -1, kernely)
            return cv2.add(img_robertsx, img_robertsy)
        elif filter_type == "Edge - Laplacian":
            output = cv2.Laplacian(gray, cv2.CV_64F)
            return np.uint8(np.clip(np.abs(output), 0, 255))
        elif filter_type == "Edge - Laplacian of Gaussian":
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            output = cv2.Laplacian(blur, cv2.CV_64F)
            return np.uint8(np.clip(np.abs(output), 0, 255))
        elif filter_type == "Edge - Canny":
            return cv2.Canny(gray, t1, t2)

    def apply_hough_transform(self, transform_type, gray):
        """Circles and Lines Detection"""
        img_copy = self.image_rgb
        if transform_type == "Lines":
            # Apply Edge Detection to Gray Scaled Image
            edges = cv2.Canny(self.gray, 50, 150)
            # Find Lines in Edge-Detected Image
            lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10) # Probabilistic Hough -> Return Line Endpoints directly
            # lines = cv2.HoughLines(edges) # Return Lines in Polar Coordinatess (rho, theta) 
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return img_copy
        elif transform_type == "Circles":
            # Second : Add Blur
            img_blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
            circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                      param1=50, param2=30, minRadius=10, maxRadius=100)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Draw circle outline
                    cv2.circle(img_copy, (x, y), r, (0, 255, 0), 2)
                    # Draw center
                    cv2.circle(img_copy, (x, y), 2, (0, 0, 255), 3)
            return img_copy

    def apply_morphological_operation(self, operation, gray, k_size):
        kernel = np.ones((k_size, k_size), np.uint8)
        if operation == "Dilation":
            return cv2.dilate(self.gray, kernel, iterations=1)
        elif operation == "Erosion":
            return cv2.erode(self.gray, kernel, iterations=1)
        elif operation == "Open":
            return cv2.morphologyEx(self.gray, cv2.MORPH_OPEN, kernel)
        elif operation == "Close":
            return cv2.morphologyEx(self.gray, cv2.MORPH_CLOSE, kernel)

    def setup_tabs(self):
        tabs = st.tabs(["Convert Image", "Add Noise", "Blur", "Point Transforms", 
                                                            "Local Transforms", "Global Transforms", "Morphological"])

        with tabs[0]:
            st.subheader("Converter")
            converter = st.selectbox("Convert into ", ["RGB", "Gray Scale"])
            converted_img = self.convert_image(converter)
            st.image(converted_img, caption=f"{converter} Image", use_container_width=True)
            # Download button
            if st.button("Save Converted Image"):
                filepath = self.save_image(converted_img, f"{converter} Converted Image")
                st.success(f"Image saved to {filepath}")

        with tabs[1]:
            st.subheader("Add Noise")
            noise_type = st.selectbox("Add Noise by ", ["Gaussian Noise", "Salt and Pepper Noise", "Poisson Noise"])
            if noise_type == "Gaussian Noise":
                mean = st.slider("Mean", 0.0, 1.0, 0.01)
                var = st.slider("Variance", 0.001, 0.1, 0.01)
                noised_img = self.add_gaussian_noise(self.image_rgb, mean, var)
            elif noise_type == "Salt and Pepper Noise":
                amount = st.slider("Amount", 0.001, 0.1, 0.01)
                s_vs_p = st.slider("Salt vs Pepper", 0.0, 1.0, 0.5)
                noised_img = self.add_salt_pepper_noise(self.image_rgb, amount, s_vs_p)
            elif noise_type == "Poisson Noise":
                noised_img = self.add_poisson_noise(self.image_rgb)
            st.image(noised_img, caption=f"Add Noise by {noise_type}", use_container_width=True)
            # Download button
            if st.button("Save Noised Image"):
                filepath = self.save_image(noised_img, f"{noise_type} image")
                st.success(f"Image saved to {filepath}")

        with tabs[2]:
            st.subheader("Blurring")
            blur_type = st.selectbox("Blur image by ", ["Average Blur", "Gaussian Blur", "Median Blur"])
            k = st.slider("Kernel Size", min_value=1, max_value=25, step=1 if blur_type == "Average Blur" else 2, value=5)
            blurred_img = self.blur_image(blur_type, k)
            st.image(blurred_img, caption=f"Image blurred by {blur_type}", use_container_width=True)
            # Download button
            if st.button("Save Blurred Image"):
                filepath = self.save_image(blurred_img, f"{blur_type} image")
                st.success(f"Image saved to {filepath}")

        with tabs[3]:
            st.subheader("Point Transform")
            subtask = st.selectbox("Choose Task", ["Brightness & Contrast", "Histogram", "Histogram Equalization"])
            if subtask == "Brightness & Contrast":
                alpha = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0)
                beta = st.slider("Brightness (beta)", -100, 100, 0)
                adjusted = self.adjust_brightness_contrast(alpha, beta)
                st.image(adjusted, caption="Brightness & Contrast Adjusted", use_container_width=True)
                if st.button("Save Adjusted Image"):
                    filepath = self.save_image(adjusted, f"Adjusted_image")
                    st.success(f"Image saved to {filepath}")
            elif subtask == "Histogram":
                fig = self.compute_histogram()
                st.pyplot(fig)
                # Download button
                if st.button("Save Histogram of Image"):
                    filepath = self.save_image(fig, f"Hitogram")
                    st.success(f"Image saved to {filepath}")
            elif subtask == "Histogram Equalization":
                equalized = self.equalize_histogram()
                st.image(equalized, caption="Histogram Equalized", use_container_width=True)
                # Download button
                if st.button("Save Equalized Image"):
                    filepath = self.save_image(equalized, f"Equalized_image")
                    st.success(f"Image saved to {filepath}")

        with tabs[4]:
            st.subheader("Local Filtering & Edge Detection")
            filter_type = st.selectbox("Choose Task", [
                "Low Pass Filter (Blur)", "High Pass Filter (Sharpen)", "Edge - Sobel", 
                "Edge - Prewitt", "Edge - Roberts", "Edge - Laplacian", 
                "Edge - Laplacian of Gaussian", "Edge - Canny"
            ])
            k = st.slider("Kernel Size", min_value=1, max_value=15, step=2, value=5) if filter_type in ["Low Pass Filter (Blur)"] else 5
            t1 = st.slider("Canny Threshold 1", 50, 150, 50) if filter_type == "Edge - Canny" else 50
            t2 = st.slider("Canny Threshold 2", 100, 300, 150) if filter_type == "Edge - Canny" else 150
            output = self.apply_local_filter(filter_type, self.gray, k, t1, t2)
            st.image(output, caption=filter_type, use_container_width=True)
            # Download button
            if st.button("Save Filtered Image"):
                filepath = self.save_image(output, f"{filter_type} image")
                st.success(f"Image saved to {filepath}")

        with tabs[5]:
            st.subheader("Hough Transform")
            transform_type = st.selectbox("Choose Type", ["Lines", "Circles"])
            transformed_img = self.apply_hough_transform(transform_type, self.gray)
            st.image(transformed_img, caption=f"Detected {transform_type}", use_container_width=True)
            # Download button
            if st.button("Save Transformed Image"):
                filepath = self.save_image(transformed_img, f"Detected_{transform_type}_image")
                st.success(f"Image saved to {filepath}")

        with tabs[6]:
            st.subheader("Morphological Operations")
            operation = st.selectbox(label = "Select Task",options= ["Dilation", "Erosion", "Open", "Close"])
            k_size = st.slider("Kernel Size", 1, 10, 3)
            result = self.apply_morphological_operation(operation, self.gray, k_size)
            st.image(result, caption=f"{operation} Result", use_container_width=True)
            # Download button
            if st.button("Save Image"):
                filepath = self.save_image(result, f"Image_with_{operation}")
                st.success(f"Image saved to {filepath}")

if __name__ == "__main__":
    # Apply Singleton Design Pattern To Ensure Only one Intance Created.
    processor = ImageProcessor()
    # p = ImageProcessor()
