# Brachial-Plexus-Nerve-Trunks-Recognition
### CNN_model

This folder contains all the information required to create a trained CNN model.

### Training the Model

1. Open `KR260_Brachial_Plexus_Nerve_Trunks_Recognition_in_Ultrasound_Images.ipynb` in Google Colab.
2. Run all cells to create and train the model.
3. The trained model will be saved as `ins+res.h5`.

### quantize

This folder contains the code to quantize the trained model.

- **quantize.py**: A script to quantize the `ins+res.h5` model.

### compile

This folder contains a script to compile the model.

- **compile.sh**: A shell script to compile the model.


### Application_codes

This folder contains the client and server code for the application, as well as the `kr260.xmodel` file.

- **client.py**: Client code that runs on the KR260 board.
- **kr260.xmodel**: The compiled model file for the KR260 board.
- **server.py**: Server code that runs on a laptop, captures input video, and places it in the same location as `server.py`.