package neuralnetwork

import (
	"Go-Machine-Learning/datasets/boston"
	"fmt"
)

func MLPRegressionExample() {

	X, y := boston.LoadBostonData()

	fmt.Println(X.Dims())
	fmt.Println(y.Dims())
}
