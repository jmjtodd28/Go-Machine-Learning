package models

import (
	"Go-Machine-Learning/utils"
	"math"
)

//Allows all the linear model types to use the RSquared error
type Predictor interface {
	Predict(X *utils.Matrix) (float64, error)
}

func RSquared(model Predictor, X, y *utils.Matrix) float64 {

	// First find the total sum of squares
	// To do this find the mean of the actual values yBar
	yTotal := 0.0

	for _, a := range y.Data {
		yTotal += a
	}

	yBar := yTotal / float64(len(y.Data))

	SSTot := 0.0

	for i := range y.Data {
		SSTot += math.Pow((y.Data[i] - yBar), 2)
	}

	// Next find the residual sum of squares SSRES
	SSRes := 0.0

	for i := range X.Rows {
		predict := utils.CreateMatrix(1, len(X.Row(i).Data), X.Row(i).Data)
		yHat, _ := model.Predict(predict)
		SSRes += math.Pow((y.Data[i] - yHat), 2)
	}

	RSquared := 1 - (SSRes / SSTot)

	return RSquared
}
