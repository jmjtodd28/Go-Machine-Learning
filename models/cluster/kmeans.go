package cluster

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Kmeans struct {
	NClusters int
	MaxIter   int
	NRuns     int
	Centers   *mat.Dense
	Inertia   float64
}

func NewKMeans() *Kmeans {
	return &Kmeans{
		NClusters: 8,
		MaxIter:   300,
		NRuns:     10,
	}
}

// calculates the euclidean distance between two points
func Euclidean(a mat.Vector, b mat.Vector) float64 {
	distance := 0.0
	_, nFeatures := a.Dims()

	for i := range nFeatures {
		distance += math.Pow(a.At(0, i)-b.At(0, i), 2)
	}

	return math.Sqrt(distance)
}

// Calculates the centers and stores them in k.Centers
func (k *Kmeans) Fit(X *mat.Dense) {
	//Init the Centers
	nSamples, nFeatures := X.Dims()
	k.Centers = mat.NewDense(k.NClusters, nFeatures, nil)

	row := make([]float64, nFeatures)
	for i := range k.NClusters {
		mat.Row(row, i, X)
		k.Centers.SetRow(i, row)
	}

	for range k.MaxIter {

		//for each sample, find its nearest center
		// count the number of times a center appears to help compute average for new centers
		centerCounts := make([]int, k.NClusters)
		// sum the points for each center so we can find average after
		centerSums := mat.NewDense(k.NClusters, nFeatures, make([]float64, k.NClusters*nFeatures))

		for i := range nSamples {
			nearestDist := math.Inf(1)
			nearestCenter := -1

			//find the closest center
			for j := range k.NClusters {
				dist := Euclidean(X.RowView(i), k.Centers.RowView(j))
				if dist < nearestDist {
					nearestCenter = j
					nearestDist = dist
				}
			}
			centerCounts[nearestCenter] += 1

			//add the point to the total for the coresponding center
			for f := range nFeatures {
				val := (centerSums.At(nearestCenter, f)) + X.At(i, f)
				centerSums.Set(nearestCenter, f, val)
			}
		}

		//fmt.Printf("centerCounts: %v\n", centerCounts)
		//fmt.Printf("centerSums: %v\n", centerSums)

		//compute the new average centers using sums and counts for each center
		rows, cols := k.Centers.Dims()
		for r := range rows {
			for c := range cols {
				val := centerSums.At(r, c) / float64(centerCounts[r])
				k.Centers.Set(r, c, val)
			}
		}

		//fmt.Printf("k.Centers: %v\n", k.Centers)
	}

}

// Calculates predicted classes based on the calculated centers
func (k *Kmeans) Predict(X *mat.Dense) *mat.Dense {
	return X
}
