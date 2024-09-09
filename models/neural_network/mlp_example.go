package neuralnetwork

import (
	"Go-Machine-Learning/dataSets/mnist"
	"Go-Machine-Learning/utils"
)

func MLPExample() {

	X, y := mnist.LoadMnistTrain()

	y = utils.OneHotEncodeDense(10, y)

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

	mlp := NewMultiLayerPerceptron()

	mlp.Arch = []int{784, 10, 10, 10}
	mlp.Epochs = 100
	mlp.BatchSize = 128
	mlp.LearningRate = 1
	mlp.Activation = "sigmoid"

	mlp.Train(X, y)

}
