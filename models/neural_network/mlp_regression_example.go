package neuralnetwork

import (
	"Go-Machine-Learning/datasets/boston"
	"Go-Machine-Learning/preprocessing"
	"fmt"
)

func MLPRegressionExample() {

	X, y := boston.LoadBostonData()

	fmt.Println(X.Dims())
	fmt.Println(y.Dims())

	ss := preprocessing.NewStandardScaler()
	ss.FitTransform(X)

	mlp := NewMultiLayerPerceptron()

	mlp.IsClassifier = false
	mlp.Arch = []int{13, 20, 20, 1}
	mlp.Epochs = 100
	mlp.LossFunction = "MSELoss"
	mlp.LearningRate = 0.0001

	mlp.Train(X, y, nil, nil)
}
