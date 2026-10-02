# Emotion-Controlled Tic-Tac-Toe with CNN Transfer Learning

This project uses a Convolutional Neural Network (CNN) to recognize facial expressions and turn them into moves in a Tic-Tac-Toe game. Instead of choosing a row and column with normal keyboard input, the player can use their webcam and make different facial expressions.

The project also explores **transfer learning** by taking the trained facial-expression model and reusing its learned features for a second image-classification task. The transfer-learning model is compared against another model whose learned weights are randomized.

## Project Overview

The project has three main parts:

1. **Train a CNN to recognize facial expressions**
2. **Use the trained model to control Tic-Tac-Toe with a webcam**
3. **Test transfer learning on a separate image-classification dataset**

The facial-expression classifier recognizes three categories:

- Neutral
- Happy
- Surprise

Each expression corresponds to a row or column of the Tic-Tac-Toe board. The player first makes an expression to select a row, then makes another expression to select a column.

---

## Basic CNN Model

The main neural network is defined in `basic_model.py`.

The model is a CNN made of three convolution blocks followed by fully connected layers.

### Architecture

Each convolution block contains:

```text
Conv2D
↓
Batch Normalization
↓
Max Pooling
```

The number of filters increases as the network gets deeper:

```text
32 filters
↓
64 filters
↓
128 filters
```

After the convolution layers, the extracted image features are flattened and passed into a dense layer.

```text
Flatten
↓
Dense Layer - 128 neurons
↓
Dropout - 40%
↓
Softmax Output
```

The final Softmax layer contains one output for each emotion category.

The dense layer also uses L2 regularization:

```python
kernel_regularizer=regularizers.l2(1e-4)
```

This helps reduce overfitting by discouraging the network from relying too heavily on individual weights.

Dropout is also used to randomly disable 40% of the neurons during training, which provides another form of regularization.

### Model Compilation

The network is trained using:

```text
Optimizer: RMSprop
Learning Rate: 0.001
Loss Function: Categorical Crossentropy
Metric: Accuracy
```

Categorical crossentropy is appropriate because the model is choosing between multiple image classes.

---

## Dataset Preprocessing

Dataset loading is handled in `preprocess.py`.

The program uses Keras:

```python
image_dataset_from_directory()
```

to automatically load images and determine their labels from the directory structure.

The facial-expression dataset is divided into:

```text
Training Data
Validation Data
Test Data
```

The training and validation data are created from the training directory using a validation split and a fixed random seed:

```python
seed=47
```

Using a fixed seed makes the train/validation split reproducible.

Images are automatically resized to the dimensions stored in:

```python
image_size
```

from `config.py`.

Labels use categorical encoding because the CNN has multiple output classes.

The test dataset has:

```python
shuffle=False
```

so the ordering of predictions stays consistent when producing evaluation results such as the confusion matrix.

---

## Training the Facial Expression Model

The main training process is located in `train.py`.

The model is trained for:

```python
epochs = 15
```

The program performs the following steps:

```text
Load and preprocess the dataset
        ↓
Create the BasicModel
        ↓
Display the model architecture
        ↓
Train for 15 epochs
        ↓
Evaluate on the test dataset
        ↓
Generate a confusion matrix
        ↓
Save the trained model
        ↓
Plot training statistics
```

The model is saved twice.

One version receives a timestamped filename inside the `results` directory:

```text
results/basic_model_15_epochs_timestamp_XXXXXXXX.keras
```

A second copy is saved as:

```text
model.keras
```

`model.keras` is important because the webcam player and transfer-learning models load this file later.

The training history is also saved as a NumPy file so the results can be loaded and analyzed later.

---

## Training Visualization

After training, `train.py` graphs:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

This makes it easier to see how the CNN changes over each epoch and identify possible overfitting.

The program also generates a confusion matrix after evaluating the test dataset. This shows which facial expressions the network recognizes correctly and which classes it tends to confuse.

---

## Webcam Tic-Tac-Toe

The webcam interaction is implemented in `player.py`.

Several different player classes are included.

### `TicTacToePlayer`

This is the base player interface.

```python
class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()
```

Every type of player implements its own version of `get_move()`.

### `UserInputPlayer`

Allows a player to manually type a row and column.

Example:

```text
Enter x y: 1 2
```

### `RandomPlayer`

Finds every empty position on the board and randomly selects one.

This can be used as a simple computer opponent.

### `UserWebcamPlayer`

This is the main feature of the project.

Instead of entering coordinates manually, the user selects positions using facial expressions detected by the trained CNN.

The process is:

```text
Open Webcam
     ↓
Capture Facial Expression
     ↓
Crop Image
     ↓
Convert Image to Grayscale
     ↓
Resize Image
     ↓
Convert Back to RGB
     ↓
Normalize Pixel Values
     ↓
CNN Prediction
     ↓
Emotion Classification
     ↓
Tic-Tac-Toe Position
```

---

## Webcam Image Processing

When the webcam opens, frames are first converted to grayscale.

```python
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

The program then crops the frame into a square so that the image has equal width and height.

The user presses **Enter** when they are ready to capture their expression.

Before sending the captured frame into the CNN, the program resizes it:

```python
img = cv2.resize(img, image_size)
```

Because the CNN expects three color channels, the grayscale image is converted back to RGB:

```python
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

The pixel values are then converted from the normal `0-255` image range into the `0-1` range:

```python
img = img.astype("float32") / 255.0
```

Finally, an extra batch dimension is added:

```python
img = np.expand_dims(img, axis=0)
```

This converts one image into the format expected by Keras.

---

## Emotion Prediction

The webcam player loads the previously trained network:

```python
self.model = load_model("model.keras")
```

The model predicts probabilities for each facial-expression class.

```python
predictions = self.model.predict(img)
```

NumPy's `argmax()` selects the class with the highest probability:

```python
predicted_class = int(np.argmax(predictions))
```

The returned value is:

```text
0, 1, or 2
```

corresponding to one of the three facial expressions.

That value becomes either the selected Tic-Tac-Toe row or column.

For example:

```text
Neutral   -> 0
Happy     -> 1
Surprise  -> 2
```

The program repeats the process once for the row and once for the column.

The player also has the option to type the row or column manually if they do not want to use the detected expression.

---

## Running the Game

The main game is started from `run.py`.

```python
TicTacToeGUI(RandomPlayer(), UserWebcamPlayer()).run()
```

This creates a game between:

```text
Player 1: Random computer player
Player 2: Webcam-controlled human player
```

The GUI manages the Tic-Tac-Toe board while `UserWebcamPlayer` determines the human player's moves.

---

# Transfer Learning

The second part of the project experiments with **transfer learning**.

Transfer learning takes features learned by one neural network and reuses them for a different problem.

Instead of training another network completely from scratch, the project loads:

```text
model.keras
```

and uses the facial-expression CNN as a feature extractor.

The transfer-learning dataset is loaded from:

```text
data/dogcat/train
data/dogcat/validation
data/dogcat/test
```

This makes the second task a two-category classification problem.

---

## TransferedModel

`transfered_model.py` contains the transfer-learning network.

First, the original model is loaded:

```python
base_model = models.load_model("model.keras")
```

Its weights are then frozen:

```python
base_model.trainable = False
```

This means the features learned during facial-expression training are preserved.

The original classification output is removed by taking the output of the second-to-last layer:

```python
outputs=base_model.layers[-2].output
```

A new classification head is then attached:

```text
Existing CNN Features
        ↓
Dense Layer - 64 neurons
        ↓
New Softmax Output
```

For the dog/cat experiment, the Softmax output contains two categories.

Only the newly added classification layers are trained while the original feature-extraction weights remain frozen.

The transfer model uses:

```text
Optimizer: RMSprop
Learning Rate: 0.0001
Loss: Categorical Crossentropy
Metric: Accuracy
```

The smaller learning rate allows the new classification head to train more gradually.

---

## RandomModel

`random_model.py` provides a comparison for the transfer-learning experiment.

It starts by loading the same original facial-expression model:

```python
base_model = models.load_model("model.keras")
```

However, instead of preserving the learned weights, `_randomize_layers()` replaces them with randomly generated values.

For every layer that contains weights:

```python
rand_init = np.random.normal(
    loc=0.0,
    scale=0.05,
    size=shape
)
```

The random values follow a normal distribution centered around zero.

This removes the useful features learned during facial-expression training.

A new classification head is then added:

```text
Randomized CNN
      ↓
Dense Layer - 64 neurons
      ↓
Softmax Output
```

Unlike `TransferedModel`, this implementation leaves the randomized base model trainable.

The purpose of this model is to provide a comparison against the transferred model. If the transferred network learns the new problem more effectively, it provides evidence that features learned during the original task were useful for the second task.

---

## Comparing Transfer Learning and Random Initialization

`train_transfer.py` trains both models on the second dataset.

The models are:

```python
'transfered_model': TransferedModel
'random_model': RandomModel
```

Each network is trained for:

```python
epochs = 3
```

For both models, the program:

1. Builds the model
2. Prints its architecture
3. Trains it
4. Evaluates it on the test dataset
5. Generates a confusion matrix

The validation accuracy of the two models is then plotted on the same graph.

This provides a visual comparison between:

```text
Previously Learned Features
            vs.
Randomly Initialized Features
```

and demonstrates the effect of transfer learning.

---

# File Breakdown

### `basic_model.py`

Defines the CNN used for facial-expression classification.

### `preprocess.py`

Loads and prepares the facial-expression dataset and the secondary dog/cat dataset.

### `train.py`

Trains, evaluates, saves, and graphs the facial-expression CNN.

### `player.py`

Contains the different Tic-Tac-Toe player types, including the webcam-controlled player.

### `run.py`

Starts the Tic-Tac-Toe game using a random opponent and webcam-controlled player.

### `transfered_model.py`

Reuses the trained CNN's learned features for a new classification problem.

### `random_model.py`

Randomizes the original CNN's weights to provide a comparison against transfer learning.

### `train_transfer.py`

Trains and compares the transfer-learning and randomized models.

---

# Technologies Used

The project uses:

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

TensorFlow/Keras are used to create and train the neural networks, OpenCV handles webcam capture and image processing, NumPy handles prediction processing and weight randomization, and Matplotlib visualizes the training results.

---

# General Program Flow

The entire project can be summarized as:

```text
Facial Expression Images
          ↓
     Preprocessing
          ↓
        CNN
          ↓
  Emotion Classification
          ↓
     model.keras
       ↙       ↘
      ↓         ↓
Tic-Tac-Toe   Transfer Learning
Webcam Game         ↓
               Dog/Cat Dataset
                    ↓
           Compare Transferred
           vs Random Weights
```

The project therefore demonstrates both a practical application of computer vision and an experiment involving transfer learning.

---

# Running the Project

First, install the required Python packages:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

Train the facial-expression classifier:

```bash
python train.py
```

This creates:

```text
model.keras
```

After the model has been trained, run the webcam-controlled game:

```bash
python run.py
```

To perform the transfer-learning experiment:

```bash
python train_transfer.py
```

The dog/cat dataset should be organized as:

```text
data/
└── dogcat/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── validation/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

---
