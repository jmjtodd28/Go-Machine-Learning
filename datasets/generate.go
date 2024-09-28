package datasets

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Creates Gaussian blobs for clustering around the centers provided
func MakeBlobs(nSamples, nFeatures int, centers *mat.Dense, clusterStd float64) (*mat.Dense, *mat.Dense) {


  centerCoords := mat.DenseCopyOf(centers)
	nCenters, _ := centerCoords.Dims()

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	mu := make([]float64, nFeatures)
	sigma := mat.NewSymDense(nFeatures, nil)

	for i := 0; i < nFeatures; i++ {
		sigma.SetSym(i, i, 1)
	}

	sigma.ScaleSym(clusterStd*clusterStd, sigma)
	normal, _ := distmv.NewNormal(mu, sigma, nil)

	for sample := 0; sample < nSamples; sample++ {
		cluster := rand.Intn(nCenters)
		y.Set(sample, 0, float64(cluster))
		normal.Rand(X.RawRowView(sample))
		floats.Add(X.RawRowView(sample), centerCoords.RawRowView(cluster))
	}

	return X, y
}
