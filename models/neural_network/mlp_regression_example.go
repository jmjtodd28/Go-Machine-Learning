package neuralnetwork

import (
	"Go-Machine-Learning/datasets/boston"
	"fmt"
)

func MLPRegressionExample() {

	X, y := boston.LoadBostonData()

	fmt.Println(X.Dims())
	fmt.Println(y.Dims())

  mlp := NewMultiLayerPerceptron()

  mlp.IsClassifier = false
  mlp.Arch = []int{13, 10, 10, 1}
  mlp.Epochs = 1000
  mlp.LossFunction = "MSELoss"
  mlp.LearningRate = 0.01

  mlp.Train(X, y)
}
