# Emo-Flask-Repo: Emotion Recognition API

![Emotion Recognition API Banner](https://placehold.co/800x400/2980b9/ffffff/png?text=Emotion+Recognition+API)

This repository contains the backend service for the EmoTune mobile application. It's a **Flask API** designed to receive an image and return a predicted emotion based on facial expression analysis. The service uses a pre-trained deep learning model to perform real-time emotion detection.

This API is intended to be deployed on a cloud platform like **Render** and serves as the brain for the [**EmoTune Flutter App**](https://github.com/your-username/EmoTune).

---

## ✨ Features

-   **RESTful API**: A simple `/predict` endpoint to handle emotion recognition requests.
-   **Deep Learning Model**: Utilizes a trained PyTorch model (`model.pth`) for accurate facial emotion classification.
-   **Image Preprocessing**: Includes necessary transformations to prepare images for the model.
-   **Deployment Ready**: Comes with a `render.yaml` file for easy deployment on Render.

---

## 🛠️ Tech Stack

-   **Framework**: Flask
-   **Machine Learning**: PyTorch, Torchvision, OpenCV
-   **Deployment**: Render, Gunicorn

---

## 📁 Project Structure

```
EmotionRecognitionFlask/
├── app.py              # Main Flask application file
├── model.py            # Model class definition (if separated)
├── model.pth           # The pre-trained PyTorch model weights
├── requirements.txt    # Python dependencies for pip
├── render.yaml         # Deployment configuration for Render
├── deploy.prototxt     # Caffe model definition for face detection
├── res10_300x300_ssd_iter_140000.caffemodel # Caffe model weights
└── ... (other helper scripts like augmentation.py)
```

---

## 🚀 Getting Started

Follow these instructions to get the backend server running on your local machine for development and testing.

### Prerequisites

-   Python 3.8+ and pip
-   Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ChinmayBansal010/EmotionRecognitionFlask.git
    cd emo-flask-repo
    ```

2.  **Set up a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application Locally

1.  **Start the Flask Server:**
    ```sh
    flask run
    ```
    The backend server will start and be accessible at `http://127.0.0.1:5000`.

---

##  Usage (API Endpoint)

The API has a single endpoint for making predictions.

### `POST /predict`

Send a POST request with an image file to get the emotion prediction.

-   **URL**: `/predict`
-   **Method**: `POST`
-   **Body**: `multipart/form-data` with a key `file` containing the image.

**Example using cURL:**
```sh
curl -X POST -F "file=@/path/to/your/image.jpg" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
```

**Success Response (200 OK):**
```json
{
  "emotion": "Happy"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "No file part"
}
```

---

## 🚢 Deployment

This project is configured for easy deployment on **Render**. Simply link this GitHub repository to a new Web Service on Render, and it will automatically use the `render.yaml` file to build and deploy the application.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or the API, please fork the repo and create a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
