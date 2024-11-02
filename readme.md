# Mask Detector

This project is a mask detection system that uses a pre-trained deep learning model to detect whether a person is wearing a mask, wearing a mask incorrectly, or not wearing a mask at all. The system uses OpenCV for real-time video capture and face detection, and TensorFlow for mask detection.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Pandas

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/mask-detector.git
    cd mask-detector
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the pre-trained model from the following link and place it in the main folder:
    [Download Model](https://drive.google.com/file/d/1nkC_eUSP4yV91Y329MpnA92mPhA3pcFM/view?usp=sharing)

## Usage

1. Ensure you have a webcam connected to your system.

2. Run the Jupyter Notebook:
    ```sh
    jupyter notebook maskDetector.ipynb
    ```

3. Execute the cells in the notebook to start the mask detection.

## How It Works

1. **Loading the Model**: The pre-trained model is loaded from `bestModel.keras`.
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('bestModel.keras')
    ```

2. **Face Detection**: The Haar Cascade classifier is used to detect faces in the video stream.
    ```python
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ```

3. **Mask Detection**: The detected faces are passed to the model to predict mask status.
    ```python
    def detect_face_mask(img):
        y_pred = model.predict(img.reshape(1, 224, 224, 3))
        return y_pred
    ```

4. **Displaying Results**: The results are displayed on the video stream with labels and bounding boxes.
    ```python
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    ```

## Example Output

![Example Output](example_output.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)