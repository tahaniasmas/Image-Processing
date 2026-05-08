# Custom-IPT — Image Processing Toolbox

## Overview

**Custom-IPT** is a desktop-based image processing application developed in Python using **Tkinter**, **NumPy**, **Pillow**, and **Matplotlib**.  
It provides a modern graphical interface for performing fundamental image processing operations without relying on OpenCV or external image-processing libraries.

The project was designed for educational and engineering purposes to demonstrate the implementation of classic image processing algorithms from scratch.

---

# Features

## Image Management

- Open image files (`PNG`, `JPG`, `BMP`, `TIFF`, `GIF`)
- Save processed images
- Reset to original image
- Undo / Redo support

---

# Point-to-Point Transformations

## Grayscale Conversion

Convert RGB images into grayscale using luminance transformation.

## Contrast & Brightness Adjustment

Apply:

\[
g(x,y) = \alpha f(x,y) + \beta
\]

Where:
- `α` controls contrast
- `β` controls brightness

## Histogram Equalization

Enhance image contrast using cumulative histogram distribution.

## Otsu Thresholding

Automatic binary threshold selection based on inter-class variance maximization.

---

# Spatial Filtering

## Smoothing Filters

- Mean Filter
- Gaussian Filter
- Median Filter

## Edge Detection

- Sobel Operator
- Prewitt Operator
- Laplacian Filter

## Image Sharpening

- Unsharp Masking

---

# Morphological Operations

The toolbox supports binary morphology operations:

- Erosion
- Dilation
- Opening
- Closing
- Skeletonization (Zhang-Suen algorithm)

---

# Analysis Tools

## Line Profile

Draw a line across the image and visualize pixel intensity variations.

## Distance Measurement

Measure Euclidean distance between two selected points.

## Histogram Visualization

Display grayscale or RGB histograms dynamically.

## Image Statistics

Compute:
- Minimum intensity
- Maximum intensity
- Mean intensity
- Standard deviation
- Image dimensions

---

# Technologies Used

- Python 3
- Tkinter
- NumPy
- Pillow (PIL)
- Matplotlib

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/custom-ipt.git
cd custom-ipt
```

## 2. Install Dependencies

```bash
pip install numpy pillow matplotlib
```

## 3. Run the Application

```bash
python main.py
```

---

# Project Structure

```text
custom-ipt/
│
├── main.py
├── README.md
└── assets/
```

---

# Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + O | Open Image |
| Ctrl + S | Save Image |
| Ctrl + Z | Undo |
| Ctrl + Y | Redo |

---

# Algorithms Implemented From Scratch

This project intentionally implements several image processing algorithms manually using NumPy:

- 2D convolution
- Histogram computation
- Histogram equalization
- Otsu thresholding
- Gaussian kernel generation
- Sobel operator
- Prewitt operator
- Laplacian filtering
- Morphological operations
- Zhang-Suen skeletonization

No OpenCV image processing functions are used.

---

# User Interface

The application contains:
- Interactive image canvas
- Dynamic histogram panel
- Analysis graphs
- Real-time cursor information
- Modern dark-themed interface

---

# Educational Objectives

This toolbox is intended for:
- Digital Image Processing courses
- Engineering projects
- Algorithm visualization
- Research demonstrations
- Learning low-level image processing techniques

---

# Future Improvements

Possible future enhancements:
- FFT / Frequency domain filtering
- Canny edge detection
- Image segmentation
- ROI selection tools
- Zoom & pan support
- GPU acceleration
- Real-time webcam processing
- Batch processing

---

# Author

Developed as part of an Image Processing / AI Engineering project.

---

# License

This project is released under the MIT License.
