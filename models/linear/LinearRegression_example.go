package models

import (
	"Go-Machine-Learning/utils"
	"fmt"
)

func ExampleLinearRegression() {

	// Training data y = 1 + 2*x1 + x2
	X := utils.CreateMatrix(4, 2, []float64{1, 2, 3, 4, 5, 6, 10, 5})
	y := utils.CreateMatrix(4, 1, []float64{5, 11, 17, 26})

	//Setting up hyperparameters
	lr := NewLinearRegression()

	// Fit the model
	lr.Fit(X, y)

	// Get the R-Squared value of the model
	fmt.Println("R-Squared Error", RSquared(lr, X, y))

	//Get the coefficients and the bias, bias should be close to 1 and the coeffs = [2, 1]
	fmt.Println(" Weights: ", lr.Coeffs)

	//Make a prediciton based on the fitted model, this example should return something close to 14 = 1 + 2x4 + 5
	predict := utils.CreateMatrix(1, 2, []float64{4, 5})
	yHat, err := lr.Predict(predict)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("Prediction:", yHat)
	}

}
