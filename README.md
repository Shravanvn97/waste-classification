🌍 Smart Waste Classification Using AI

An intelligent waste classification system that uses deep learning and computer vision to automatically detect and categorize waste into organic and recyclable materials. The system supports real-time detection using images and webcam and provides recycling guidance to improve waste management efficiency.

🚀 Project Overview

Waste segregation is essential for effective recycling and environmental sustainability. This project develops an AI-powered solution that detects waste objects using YOLO and classifies them using CNN-based models. The system is deployed through a Streamlit web interface for interactive usage.

🎯 Features

Real-time waste detection using webcam

Multi-object detection with bounding boxes

Organic vs non-organic classification

Material classification (plastic, paper, glass, metal, etc.)

Recycling recommendations

Interactive dashboard

User-friendly interface

🧠 Models Used

Convolutional Neural Network (CNN) for classification

Transfer Learning for improved performance

YOLO (You Only Look Once) for object detection

Supervised learning with labeled datasets

🛠 Technologies

Python

TensorFlow / Keras

YOLO (Ultralytics)

OpenCV

Streamlit

NumPy

Pillow

Git & GitHub

📊 Dataset

The system is trained on labeled waste image datasets containing different categories of waste materials such as plastic, paper, metal, glass, and organic waste. Images were preprocessed using resizing and normalization techniques.

⚙️ How It Works

User uploads image or uses webcam

YOLO detects waste objects

CNN model classifies waste type

System determines recyclability

Results displayed through web interface
💻 Installation

Clone the repository:

git clone https://github.com/Shravanvn97/waste-classification.git
cd smart-waste-ai

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py

Open browser:

http://localhost:8501
📈 Results

The system achieves good classification accuracy and performs real-time detection effectively, demonstrating the practical application of AI in waste management.

⚠ Limitations

Performance depends on dataset quality

Lighting conditions may affect accuracy

Requires computational resources

🔮 Future Scope

Smart bin integration

IoT-based monitoring

Mobile application

Cloud deployment

Robotic sorting systems

🎓 Conclusion

This project demonstrates how artificial intelligence can automate waste classification and improve recycling efficiency. The system can be extended to support smart city infrastructure and sustainable waste management solutions.

👨‍💻 Author

Shravan V N
