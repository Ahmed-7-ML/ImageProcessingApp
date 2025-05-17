import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    _instance = None

    def __new__(cls):
        """Ensure only one instance of ImageProcessor is created (Singleton Pattern)"""
        if cls._instance is None:
            cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Image Processor Class with an Input Image"""
        # Prevent reinitialization if instance already exists
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.image = None
        self.image_rgb = None
        self.gray = None

        # URL of the online background image
        image_url = "https://drive.google.com/uc?export=view&id=1M9WsLRAW5LSFcCRZHi6LWtrPsmF_CatU"
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
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
        self.file = st.file_uploader("Upload an Image", ["png", "jpg", "jpeg"])
        self.load_image()

    def load_image(self):
        """Accept an Input Image of Type png, jpg or jpeg"""
        if self.file is not None:
            # Convert File(Image) into a Numpy Array of Pixels
            img_arr = np.asarray(bytearray(self.file.read()), dtype=np.uint8)
            self.image = cv2.imdecode(img_arr, 1)  # BGR Image
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            st.image(self.image_rgb, caption="Uploaded Image (RGB)", use_container_width=True)
            self.setup_tabs()
        else:
            st.info("Please Upload an Image to Get Started")

    def prepare_image_download(self, image, operation):
        """Prepare the image for download as PNG bytes"""
        filename = f"{operation.lower()}.png"
        try:
            _, buffer = cv2.imencode(".png", image)
            image_bytes = buffer.tobytes()
            return image_bytes, filename, None
        except Exception as e:
            return None, None, f"Error preparing image for download: {str(e)}"

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
        noisy = np.random.poisson(image.astype(float))
        noisy = np.clip(noisy, 0, 255)
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
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased blur for noise reduction
        gray_enhanced = cv2.equalizeHist(blurred)  # Enhance contrast
        edges = cv2.Canny(gray_enhanced, 30, 100)  # Adjusted Canny thresholds

        if transform_type == "Lines":
            # Find Lines in Edge-Detected Image
            lines = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi / 260, threshold=40, minLineLength=10, maxLineGap=60)
            filtered_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if 45 < abs(angle) < 135:  # Keep near-vertical lines
                        filtered_lines.append(line)
            lines = np.array(filtered_lines) if filtered_lines else None

            # Create a copy of the original image for drawing lines
            lines_image = self.image.copy()
            # Draw detected lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines

            # Simulate Hough Accumulator Visualization
            hough_accum = np.zeros_like(gray)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hough_accum, (x1, y1), (x2, y2), 255, 1)
                hough_accum = cv2.dilate(hough_accum, np.ones((5, 5), np.uint8), iterations=2)

            return lines_image, hough_accum

        elif transform_type == "Circles":
            # Enhanced preprocessing for coin detection
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Lighter blur to preserve edges
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Otsu thresholding
            edges = cv2.Canny(thresh, 100, 200)  # Adjusted Canny thresholds for coin edges

            # Apply Hough Circle Transform with tuned parameters
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                      param1=200, param2=20,
                                      minRadius=50, maxRadius=100)  # Constrain radius based on coin size

            # Create a copy of the original image for drawing circles
            circles_image = self.image.copy()

            # Draw detected circles on the original image
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Draw the circle outline
                    cv2.circle(circles_image, (x, y), r, (0, 0, 255), 2)  # Red outline
                    # Draw the center of the circle
                    cv2.circle(circles_image, (x, y), 2, (255, 0, 0), 3)  # Blue center

            # Simulate Accumulator Visualization (approximation)
            accum = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
            if circles is not None:
                for (x, y, r) in circles:
                    cv2.circle(accum, (x, y), r, 255, 1)
                accum = cv2.dilate(accum, np.ones((5, 5), np.uint8), iterations=2)

            return circles_image, accum

    def apply_morphological_operation(self, operation, gray, k_size):
        """Apply Morphological Operations"""
        kernel = np.ones((k_size, k_size), np.uint8)
        if operation == "Dilation":
            return cv2.dilate(gray, kernel, iterations=1)
        elif operation == "Erosion":
            return cv2.erode(gray, kernel, iterations=1)
        elif operation == "Open":
            return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == "Close":
            return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    def setup_tabs(self):
        """Set up tabs for different image processing operations"""
        tabs = st.tabs(["Convert Image", "Add Noise", "Blur", "Point Transforms",
                        "Local Transforms", "Global Transforms", "Morphological"])

        with tabs[0]:
            st.subheader("Converter")
            converter = st.selectbox("Convert into ", ["RGB", "Gray Scale"])
            converted_img = self.convert_image(converter)
            st.image(converted_img, caption=f"{converter} Image", use_container_width=True)
            image_bytes, filename, error = self.prepare_image_download(converted_img, f"converted_{converter}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label=f"Download Converted Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

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
            image_bytes, filename, error = self.prepare_image_download(noised_img, f"noised_{noise_type}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label="Download Noised Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

        with tabs[2]:
            st.subheader("Blurring")
            blur_type = st.selectbox("Blur image by ", ["Average Blur", "Gaussian Blur", "Median Blur"])
            k = st.slider("Kernel Size", min_value=1, max_value=25, step=1 if blur_type == "Average Blur" else 2, value=5)
            blurred_img = self.blur_image(blur_type, k)
            st.image(blurred_img, caption=f"Image blurred by {blur_type}", use_container_width=True)
            image_bytes, filename, error = self.prepare_image_download(blurred_img, f"blurred_{blur_type}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label="Download Blurred Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

        with tabs[3]:
            st.subheader("Point Transform")
            subtask = st.selectbox("Choose Task", ["Brightness & Contrast", "Histogram", "Histogram Equalization"])
            if subtask == "Brightness & Contrast":
                alpha = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0)
                beta = st.slider("Brightness (beta)", -100, 100, 0)  # Fixed the typo here
                adjusted = self.adjust_brightness_contrast(alpha, beta)
                st.image(adjusted, caption="Brightness & Contrast Adjusted", use_container_width=True)
                image_bytes, filename, error = self.prepare_image_download(adjusted, "brightness_contrast")
                if error:
                    st.error(error)
                else:
                    st.download_button(
                        label="Download Adjusted Image",
                        data=image_bytes,
                        file_name=filename,
                        mime="image/png"
                    )
            elif subtask == "Histogram":
                fig = self.compute_histogram()
                st.pyplot(fig)
            elif subtask == "Histogram Equalization":
                equalized = self.equalize_histogram()
                st.image(equalized, caption="Histogram Equalized", use_container_width=True)
                image_bytes, filename, error = self.prepare_image_download(equalized, "histogram_equalized")
                if error:
                    st.error(error)
                else:
                    st.download_button(
                        label="Download Equalized Image",
                        data=image_bytes,
                        file_name=filename,
                        mime="image/png"
                    )

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
            image_bytes, filename, error = self.prepare_image_download(output, f"filtered_image_with_{filter_type}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label="Download Filtered Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

        with tabs[5]:
            st.subheader("Hough Transform")
            transform_type = st.selectbox("Choose Type", ["Lines", "Circles"])
            if transform_type == "Lines":
                lines_image, hough_accum = self.apply_hough_transform(transform_type, self.gray)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(self.image_rgb, caption="Original image", use_container_width=True)
                with col2:
                    st.image(hough_accum, caption="The hough", use_container_width=True, clamp=True)
                with col3:
                    st.image(lines_image, caption="Extract line segments based on Hough transform", use_container_width=True)
                image_bytes, filename, error = self.prepare_image_download(lines_image, f"hough_{transform_type.lower()}")
            elif transform_type == "Circles":
                circles_image, accum = self.apply_hough_transform(transform_type, self.gray)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(self.image_rgb, caption="Original image", use_container_width=True)
                with col2:
                    st.image(accum, caption="The hough", use_container_width=True, clamp=True)
                with col3:
                    st.image(circles_image, caption="The image circles", use_container_width=True)
                image_bytes, filename, error = self.prepare_image_download(circles_image, f"hough_{transform_type.lower()}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label="Download Transformed Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

        with tabs[6]:
            st.subheader("Morphological Operations")
            operation = st.selectbox(label="Select Task", options=["Dilation", "Erosion", "Open", "Close"])
            k_size = st.slider("Kernel Size", 1, 10, 3)
            result = self.apply_morphological_operation(operation, self.gray, k_size)
            st.image(result, caption=f"{operation} Result", use_container_width=True)
            image_bytes, filename, error = self.prepare_image_download(result, f"morph_{operation.lower()}")
            if error:
                st.error(error)
            else:
                st.download_button(
                    label="Download Morphological Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

if __name__ == "__main__":
    processor1 = ImageProcessor()
    processor2 = ImageProcessor()
    print(processor1 is processor2)  # Should print True, confirming both point to the same instance
