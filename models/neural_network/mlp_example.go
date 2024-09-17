package neuralnetwork

import (
	"Go-Machine-Learning/dataSets/mnist"
	"Go-Machine-Learning/preprocessing"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func MLPExample() {

	X, y := mnist.LoadMnistTrain()

	y = preprocessing.OneHotEncodeDense(10, y)

	//normalise data
	rows, cols := X.Dims()
	for i := range rows {
		for j := range cols {
			value := X.At(i, j) / 255
			X.Set(i, j, value)
		}
	}

	/*
		//Creating a and gate
		X := mat.NewDense(4, 2, []float64{
			0, 0,
			1, 0,
			0, 1,
			1, 1})


		y := mat.NewDense(4, 1, []float64{
			0,
			1,
			1,
			0,
		})
	*/

  // Set Hyperparameters
	mlp := NewMultiLayerPerceptron()
	mlp.Arch = []int{784, 40, 10}
	mlp.Epochs = 10
	mlp.BatchSize = 128
	mlp.LearningRate = 0.02

  //Train the model
	mlp.Train(X, y)

  //Make a prediciton of one of the samples
  _, xcols := X.Dims()
	xPredict := X.Slice(1, 3, 0, xcols).(*mat.Dense)
  prediction := mlp.Predict(xPredict)
  fmt.Printf("prediction: %v\n", prediction)

}
