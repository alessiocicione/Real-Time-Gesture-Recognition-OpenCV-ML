# Real-Time Hand Gesture Recognition System (C++/OpenCV)

This repository contains the complete source code and necessary files for a **Real-Time Hand Gesture Recognition System** based on Computer Vision and Machine Learning.

The system is designed to classify **9 distinct hand gestures** in real-time, achieving high accuracy through a high-performance hybrid C++ and Python implementation.

---

## Technology Stack and Implementation

This project utilizes a dual-language architecture, leveraging the performance of C++ for vision processing and the extensive libraries of Python for Machine Learning:

* **Computer Vision & Feature Extraction (C++):** Implemented using **C++ and OpenCV**. This module handles the high-speed execution loop, including:
    * Reading the live video feed.
    * Detecting coded markers placed on a glove.
    * Extracting **geometric features** from these markers (e.g., distances, visibility, relative positions).
* **Machine Learning (Python):** Implemented using **Python and scikit-learn**. This module is used for data handling and model development:
    * Data labeling and preparation.
    * Training multiple classifiers (SVM, Random Forest, KNN).
    * **Model Selection:** The **Support Vector Machine (SVM)** classifier was selected as the final model, achieving an **accuracy of 97.6%** on the dedicated test set.
* **Real-Time Integration:** The C++ and Python components are integrated using **pybind11**, allowing the C++ application to load and execute the Python-trained ML model seamlessly in a real-time loop.

---

## ML-Main Development and Dataset Details

This section provides specific context for the files and datasets found in the **`ML-Main/`** directory, which was used for classifier training and validation.

* **`main.py`**: Contains the **Grid Search** logic and comprehensive performance **reports** for the various classifiers tested (SVM, Random Forest, KNN).
* **`SVMtrainSave.py`**: Handles the final **export** of the chosen **SVM model** and the necessary **data scaler** for subsequent use in the C++ environment via pybind11.

### Datasets

* The most recent and utilized dataset version is structured as `gesture_*_dataset_4.csv` (where `*` is `train` or `test`).
* This dataset is located in the **`DatasetsCsv\Dataset 3`** folder.
* The older dataset, corresponding to **Dataset 0**, is labeled with the suffix `*_v1`.
* **Noise Class:** In **Dataset 2 and Dataset 3**, the **noise class (0)** was not used during training.
* *Note on Batch Files:* Do not execute the included `.bat` file; it was only used internally to prepare video files for reading due to metadata issues from the original recording device.

---

## Testing and Validation Resources

The `videos/` folder contains several video files useful for testing the real-time application.

**Important Note:** None of the videos in the `videos/` folder were used to train the current classifier model.

| File Name | Content Description | Status |
| :--- | :--- | :--- |
| `handoCompTest8.mp4` | Contains the most recent set of all 9 gestures for comprehensive testing. | **Primary Test Video** |
| `handoCompTest6.mp4` | Older test video covering the previous dataset. Gestures are valid, but marker 0000 is intentionally not visible in some frames (which the code correctly handles by discarding frames). | **Secondary Test Video** |
| `hando*.mp4` (others) | Generally contain single gestures. These were training videos from an older dataset but are reused here exclusively for testing the current model. | **Supplementary Tests** |
| `hando10.mp4` | Includes two gestures: open hand and closed hand. | **Supplementary Test** |

---

## Real-Time Execution Details

When running the application, the prediction confidence level is displayed next to the classified gesture: `conf: ...`

* **Green Confidence:** The confidence level is **75% or higher**.
* **Yellow Confidence:** The confidence level is **below 75%**.

---

## Setup and Execution

### Prerequisites

1.  C++ Compiler
2.  OpenCV Library
3.  Python 3.x
4.  Required Python Packages (scikit-learn, numpy, etc.)
5.  pybind11
