package com.mithun.cfd;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.io.File;

public class Trainer {

    // dataset and model paths
    private static final String DATASET_PATH = "data/faces/lfw-deepfunneled";
    private static final String MODEL_PATH = "models/lbph-model.xml";

    public void train() {
        try {
            File root = new File(DATASET_PATH);
            if (!root.exists() || !root.isDirectory()) {
                System.out.println("Dataset not found at: " + DATASET_PATH);
                return;
            }

            // collect person folders
            File[] personFolders = root.listFiles(File::isDirectory);
            if (personFolders == null || personFolders.length == 0) {
                System.out.println("No person folders found in dataset path: " + DATASET_PATH);
                return;
            }

            // First count total images to create arrays/buffers
            int totalImages = 0;
            for (File personFolder : personFolders) {
                File[] imgs = personFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg")
                        || name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpeg"));
                if (imgs != null) totalImages += imgs.length;
            }

            if (totalImages == 0) {
                System.out.println("No images found in dataset.");
                return;
            }

            // Prepare MatVector and labels pointer
            MatVector images = new MatVector(totalImages);
            IntPointer labelsPointer = new IntPointer(totalImages);

            int imgIndex = 0;
            int label = 0;

            for (File personFolder : personFolders) {
                File[] imgs = personFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg")
                        || name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpeg"));
                if (imgs == null) continue;

                for (File imgFile : imgs) {
                    Mat img = opencv_imgcodecs.imread(imgFile.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
                    if (img == null || img.empty()) {
                        System.out.println("Warning: couldn't read image: " + imgFile.getAbsolutePath());
                        continue;
                    }

                    // resize to consistent size (optional but recommended)
                    opencv_imgproc.resize(img, img, new Size(200, 200));

                    images.put(imgIndex, img);
                    labelsPointer.put(imgIndex, label);

                    imgIndex++;
                    if (imgIndex % 500 == 0) System.out.println("Loaded images: " + imgIndex);
                }
                label++;
            }

            System.out.println("Training images: " + imgIndex + ", labels used: " + label);

            // Create labels Mat from IntPointer
            Mat labelsMat = new Mat(imgIndex, 1, org.bytedeco.opencv.global.opencv_core.CV_32SC1, labelsPointer);

            // If fewer images were read than counted (due to read errors), shrink structures
            if (imgIndex != totalImages) {
                // rebuild MatVector and labelsMat for actual count
                MatVector realImages = new MatVector(imgIndex);
                IntPointer realLabels = new IntPointer(imgIndex);
                for (int i = 0; i < imgIndex; i++) {
                    realImages.put(i, images.get(i));
                    realLabels.put(i, labelsPointer.get(i));
                }
                images = realImages;
                labelsMat = new Mat(imgIndex, 1, org.bytedeco.opencv.global.opencv_core.CV_32SC1, realLabels);
            }

            // create recognizer
            LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();

            // train
            recognizer.train(images, labelsMat);

            // ensure model folder exists
            File modelFile = new File(MODEL_PATH);
            File parent = modelFile.getParentFile();
            if (parent != null && !parent.exists()) parent.mkdirs();

            // save model
            recognizer.save(MODEL_PATH);
            System.out.println("Training completed! Model saved at: " + MODEL_PATH);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // convenience main for testing
    public static void main(String[] args) {
        Trainer t = new Trainer();
        t.train();
    }
}
