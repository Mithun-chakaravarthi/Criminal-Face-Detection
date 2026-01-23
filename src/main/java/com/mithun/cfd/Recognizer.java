package com.mithun.cfd;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Recognizer {
    private CascadeClassifier faceCascade;
    private LBPHFaceRecognizer recognizer;

    public Recognizer(CascadeClassifier faceCascade, String modelPath) {
        this.faceCascade = faceCascade;
        this.recognizer = LBPHFaceRecognizer.create();
        try {
            recognizer.read(modelPath);
        } catch (Exception e) {
            System.err.println("Cannot read model: " + modelPath + " — run training first.");
        }
    }

    public void startCamera() {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("Cannot open camera");
            return;
        }
        Mat frame = new Mat();
        while (true) {
            if (!camera.read(frame)) continue;
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);
            RectVector faces = new RectVector();
            faceCascade.detectMultiScale(gray, faces);
            for (int i = 0; i < faces.size(); i++) {
                Rect r = faces.get(i);
                Mat face = new Mat(gray, r).clone();
                opencv_imgproc.resize(face, face, new org.bytedeco.opencv.opencv_core.Size(200,200));
                int[] label = new int[1];
                double[] conf = new double[1];
                try {
                    recognizer.predict(face, label, conf);
                    System.out.println("Detected label=" + label[0] + " confidence=" + conf[0]);
                } catch (Exception e) {
                    System.err.println("Predict failed — model missing or incompatible.");
                }
            }
            // stop process with Ctrl+C in the terminal
        }
    }
}
