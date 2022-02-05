# LSTM_NN
Prediction of a sinusoidal function using a Recursive Neural Network

The next model shows a simple model of a recursive neural network for predicting a sinusoidal function (sin(πt)). A time frame is created and split in 10000 steps, which allows us to compute the values of the sinusoidal function. The sequential model consists in a two-hidden layers network, with an LSTM layer of 16 nodes and a Dense layer of 1 (output), yielding 1169 trainable parameters. No activation functions were required, as it is a regression problem, not a classification one. Then, the model was fit through 60 epochs  with a batch size of 32, and two callbacks (ReduceLROnPlateau, EarlyStopping) to avoid overfitting and reducing running time. The following loss graph shows our model has been trained successfully. 

![Loss_LSTM](https://user-images.githubusercontent.com/96789733/152637084-494027ec-69e7-45cf-b3b0-bad5f72e61a9.png)

Once the model was trained, a prediction was performed for the first 500 time intervals. As it can be observed in the graphic, the prediction fit perfectly 
the real values, so we can state that the model was completely successful on its task.

![Prediction_LSTM](https://user-images.githubusercontent.com/96789733/152637236-5537b24f-2b24-4ac8-a7fc-31f8974cc45f.png)
