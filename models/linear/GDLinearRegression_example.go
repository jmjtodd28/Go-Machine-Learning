package models

import (
	"Go-Machine-Learning/utils"
	"fmt"
)

//simple example of linear regression using gradient descent
func ExampleGDLinearRegression() {

	// Training data y = 2*x1 + x2
	X := utils.CreateMatrix(4, 2, []float64{1, 2, 3, 4, 5, 6, 10, 5})
	y := utils.CreateMatrix(4, 1, []float64{4, 10, 16, 25})

	//Setting up hyperparameters
	glr := NewGDLinearRegression()
	glr.MaxIter = 1000
	glr.GDescentType = "batch"
  glr.earlyStopping = false

	//Control whether you want information printed to the console
	glr.Verbose = false

	// Fit the model
	glr.Fit(X, y)

	// Get the R-Squared value of the model
	fmt.Println("R-Squared Error", RSquared(glr, X, y))

	//Get the coefficients and the bias, bias should be close to 0 and the coeffs = [2, 1]
	fmt.Println("Bias:, ", glr.Bias, " Weights: ", glr.Coeffs.Data)

	//Make a prediciton based on the fitted model, this example should return something close to 13 = 0 + 2x4 + 5
	predict := utils.CreateMatrix(1, 2, []float64{4, 5})
	yHat, err := glr.Predict(predict)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("Prediction:", yHat)
	}

}
