# Criminal Face Detection

A Java-based computer vision project that detects and recognizes human faces using machine learning techniques.

---

## Features

* Face detection using **OpenCV Haar Cascade**
* Face recognition using **LBPH algorithm**
* Training with custom or LFW dataset
* Real-time recognition through webcam
* Maven-based clean project structure
* GitHub-ready lightweight repository

---

## Tech Stack

* Java
* OpenCV (JavaCPP / Bytedeco)
* Maven
* Computer Vision & Machine Learning basics

---

## Dataset

This project uses the **LFW (Labeled Faces in the Wild)** dataset.

Download from Kaggle:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

After downloading, place the dataset inside:

```
data/faces/lfw-deepfunneled
```

---

## How to Run

### 1️⃣ Build the Project

```bash
mvn clean package
```

### 2️⃣ Train the Model

```bash
java --enable-native-access=ALL-UNNAMED -jar target/criminal-face-detection-1.0-SNAPSHOT-jar-with-dependencies.jar train
```

### 3️⃣ Run Face Recognition

```bash
java --enable-native-access=ALL-UNNAMED -jar target/criminal-face-detection-1.0-SNAPSHOT-jar-with-dependencies.jar recognize
```

---

## Project Structure

```
src/
 ├── main/java/com/mithun/cfd
 │    ├── Main.java
 │    ├── Trainer.java
 │    ├── Recognizer.java
 │
resources/
 └── haarcascade_frontalface_default.xml
```

---

## Author

**Mithun Chakaravarthi**

---

## Future Improvements

* Deep learning face recognition (FaceNet / Dlib)
* Criminal database integration
* Web dashboard for monitoring
* Secure authentication system
