package neuralnetwork

import (
	"Go-Machine-Learning/dataSets/mnist"
	"Go-Machine-Learning/preprocessing"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func MLPExample() {

	XTrain, yTrain := mnist.LoadMnistTrain()
	XTest, yTest := mnist.LoadMnistTest()

	yTrain = preprocessing.OneHotEncodeDense(10, yTrain)
	yTest = preprocessing.OneHotEncodeDense(10, yTest)

	//normalise data
	rows, cols := XTrain.Dims()
	for i := range rows {
		for j := range cols {
			value := XTrain.At(i, j) / 255
			XTrain.Set(i, j, value)
		}
	}
	rows, cols = XTest.Dims()
	for i := range rows {
		for j := range cols {
			value := XTest.At(i, j) / 255
			XTest.Set(i, j, value)
		}
	}
	/*
		//Creating a and gate
		XTrain := mat.NewDense(4, 2, []float64{
			0, 0,
			1, 0,
			0, 1,
			1, 1})

		yTrain := mat.NewDense(4, 1, []float64{
			0,
			1,
			1,
			0,
		})

		yTrain = preprocessing.OneHotEncodeDense(2, yTrain)

	*/
	// Set hyperparameters
	mlp := NewMultiLayerPerceptron()
	mlp.Arch = []int{784, 40, 10}
	mlp.Epochs = 20
	mlp.BatchSize = 128
	mlp.LearningRate = 0.01
	mlp.Activation = "relu"
	mlp.IsClassifier = true

	//Train the model
	mlp.Train(XTrain, yTrain, XTest, yTest)

	//Make a prediciton of one of the samples
	_, xcols := XTrain.Dims()
	xPredict := XTrain.Slice(1, 3, 0, xcols).(*mat.Dense)
	//fmt.Printf("xPredict: %v\n", xPredict)
	prediction := mlp.Predict(xPredict)
	fmt.Printf("prediction: %v\n", prediction)

}
