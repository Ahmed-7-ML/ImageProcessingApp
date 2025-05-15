# 🎉 Image Processing App
### Developed by: Eng. Ahmed Akram Amer & Eng. Ibrahim Mohamed Hashish

A **Streamlit-based web application** for performing a wide range of **image processing operations** with a **user-friendly interface** and a customizable background image.

---

## 🚀 Features

* **📤 Image Upload:** Supports PNG, JPG and JPEG formats.
* **🎨 Image Processing Operations:**

  * Convert images to RGB or Grayscale
  * Add noise (Gaussian, Salt & Pepper, Poisson)
  * Apply blurring (Average, Gaussian, Median)
  * Brightness & Contrast adjustment
  * Histogram computation & Equalization
  * Edge detection (Sobel, Prewitt, Roberts, Laplacian, Canny, etc.)
  * Hough Transforms for line and circle detection
  * Morphological operations (Dilation, Erosion, Opening, Closing)

---

## ⚙️ Prerequisites

* Python 3.8 or higher
* A modern web browser (Chrome, Firefox, Edge, etc.)
* Git (for cloning the repository)

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmed-7-ML/ImageProcessingApp.git
cd image-processing-app
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

Ensure `requirements.txt` contains:

```txt
streamlit
opencv-python
numpy
matplotlib
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Background Image

Place `bg.jpg` in the root directory. Supported formats: JPEG, PNG.
Or update the `image_path` in `streamlit_app.py` accordingly.

---

## ▶️ Usage

Run the app using:

```bash
streamlit run streamlit_app.py
```

Access it in your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧑‍💻 How to Use

1. **Upload Image** via sidebar uploader.
2. **Choose a Tool** from tabs (e.g., "Add Noise", "Blur", "Edge Detection").
3. **Adjust Parameters** using sliders and dropdowns.
4. **View Results:** Original image on the left, processed output on the right.
5. **Toggle Theme** (Light/Dark) from the sidebar.
6. **Reset Image** anytime from the sidebar.

---

## 💡 Tips

* Use high-resolution images for better edge detection & Hough transforms.
* Make sure `bg.jpg` is valid and in the correct path.

---

## 📁 File Structure

```
image-processing-app/
├── streamlit_app.py     # Main application script
├── bg.jpg               # Background image
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

---

## ☁️ Deploy on Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/image-processing-app.git
git push -u origin main
```

### 2. Setup on Streamlit Cloud

* Sign in at [streamlit.io](https://streamlit.io) using GitHub.
* Click **"New app"**, select the repo.
* Set script path: `streamlit_app.py`
* Ensure `requirements.txt` and `bg.jpg` are present.
* Click **Deploy** 🎉

---

## 🛠️ Troubleshooting

| Problem                    | Solution                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **App doesn’t start**      | Use `streamlit run streamlit_app.py` (not `python`). Confirm dependencies and Python version (>=3.8).          |
| **Background not showing** | Ensure `bg.jpg` is in root and is a valid image. Update `image_path` if needed.                                |
| **No image output**        | Use supported formats (PNG, JPG, JPEG, GIF). Check terminal or browser console for errors. Try smaller images. |

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add your feature"
   ```
4. Push and open a Pull Request:

   ```bash
   git push origin feature/your-feature
   ```

---

## 🎉 Happy Image Processing!
