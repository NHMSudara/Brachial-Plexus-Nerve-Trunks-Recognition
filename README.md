# Brachial-Plexus-Nerve-Trunks-Recognition

### Application_codes

This folder contains the client and server code for the application, as well as the `kr260.xmodel` file.

- **client.py**: Client code that runs on the KR260 board.
- **kr260.xmodel**: The compiled model file for the KR260 board.
- **server.py**: Server code that runs on a laptop, captures input video, and places it in the same location as `server.py`.

### CNN_model

This folder contains all the information required to create a trained CNN model.

- **train_model.ipynb**: A Colab notebook that creates and trains the CNN model. The output model is named `ins+res.h5`.

### compile

This folder contains a script to compile the model.

- **compile.sh**: A shell script to compile the model.

### quantize

This folder contains the code to quantize the trained model.

- **quantize.py**: A script to quantize the `ins+res.h5` model.

## Getting Started

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/install)
- [Colab](https://colab.research.google.com/)
- [AMD KRIA KR260 Robotics Starter Kit](https://www.xilinx.com/products/som/kria/kr260-robotics-starter-kit.html)

### Training the Model

1. Open `train_model.ipynb` in Google Colab.
2. Run all cells to create and train the model.
3. The trained model will be saved as `ins+res.h5`.

### Quantizing the Model

1. Place `ins+res.h5` in the `quantize` folder.
2. Run the following command to quantize the model:
   ```sh
   python quantize.py
