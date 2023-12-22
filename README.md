# Dogs/Cats Gingivitis Estimator

The **Dogs/Cats Gingivitis Estimator** is a tool for diagnosing gingivitis in dogs and cats. It requires images of the back teeth (molars) for analysis.

## Usage

1. **Select the Target**:
    - Choose between analyzing an **image** or using the **camera**.

2. **Image Analysis**:
    - Upload an image (in PNG, JPG, JPEG, or WebP format).
    - The tool will perform analysis on the uploaded image.
    - The results will be displayed, including bounding boxes around detected areas.

3. **Interpretation**:
    - The tool classifies the gingivitis severity into four categories:
        - **GI=0**: Normal gums (green).
        - **GI=1**: Mild gingivitis (orange).
        - **GI=2**: Moderate gingivitis (red).
        - **GI=3**: Severe gingivitis (red).

## Requirements

- Python 3.6 or higher
- Streamlit
- Ultralytics YOLO 
- Other necessary libraries (Pandas, Matplotlib, PIL, NumPy, OpenCV)

## Installation

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.

## How to Run

Execute the following command in your terminal:

```bash
streamlit run gingivitis.py
```

## Acknowledgments

This tool was developed for educational purposes and should not replace professional veterinary advice.

---

Feel free to customize the README further based on additional details or specific instructions related to your project. 🐾🦷