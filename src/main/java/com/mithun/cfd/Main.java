package com.mithun.cfd;

import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Usage: train | recognize");
            return;
        }

        CascadeClassifier faceCascade = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");

        if (args[0].equals("train")) {
           // use default Trainer (no args)
Trainer trainer = new Trainer();

// if your Trainer has a train() method that uses default DATASET_PATH / MODEL_PATH:
trainer.train();

// OR if you want to pass custom paths via a simple method name 'train(String dataset, String model)':
// trainer.train(datasetPath, modelPath);

        } else if (args[0].equals("recognize")) {
            Recognizer recognizer = new Recognizer(faceCascade, "models/lbph-model.xml");
            recognizer.startCamera();

        } else {
            System.out.println("Unknown command: " + args[0]);
        }
    }
}
