package regression

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/skelterjohn/go.matrix"
)

var (
	// ErrNotEnoughData signals that there weren't enough datapoint to train the model.
	ErrNotEnoughData = errors.New("not enough data points")
	// ErrTooManyVars signals that there are too many variables for the number of observations being made.
	ErrTooManyVars = errors.New("not enough observations to to support this many variables")
	// ErrRegressionRun signals that the Run method has already been called on the trained dataset.
	ErrRegressionRun = errors.New("regression has already been run")
)

// Regression is the exposed data structure for interacting with the API
type Regression struct {
	Names             Describe        `json:"names"              msgpack:"names"`
	Data              []*DataPoint    `json:"data"               msgpack:"data"`
	CoeffMap          map[int]float64 `json:"coeff_map"          msgpack:"coeff_map"`
	R2                float64         `json:"r_2"                msgpack:"r_2"`
	Varianceobserved  float64         `json:"varianceobserved"   msgpack:"varianceobserved"`
	VariancePredicted float64         `json:"variance_predicted" msgpack:"variance_predicted"`
	Initialised       bool            `json:"initialised"        msgpack:"initialised"`
	Formula           string          `json:"formula"            msgpack:"formula"`
	Crosses           []featureCross  `json:"crosses"            msgpack:"crosses"`
	HasRun            bool            `json:"has_run"            msgpack:"has_run"`
	LastTrained       time.Time       `json:"last_trained"       msgpack:"last_trained"`
	Threshold         time.Duration   `json:"threshold"          msgpack:"threshold"`
}

// DataPoint basic unit used for regression
type DataPoint struct {
	Observed  float64   `json:"observed"  msgpack:"observed"`
	Variables []float64 `json:"variables" msgpack:"variables"`
	Predicted float64   `json:"predicted" msgpack:"predicted"`
	Error     float64   `json:"error"     msgpack:"error"`
}

// Describe adds optional context to the regression being done
type Describe struct {
	Obs  string         `json:"obs"  msgpack:"obs"`
	Vars map[int]string `json:"vars" msgpack:"vars"`
}

// DataPoints is a slice of pointer dataPoints.
// This type allows for easier constuction of training data points.
type DataPoints []*DataPoint

// DataPointWrapper creates a well formed *datapoint used for training.
func DataPointWrapper(obs float64, vars []float64) *DataPoint {
	return &DataPoint{Observed: obs, Variables: vars}
}

// Predict updates the "Predicted" value for the inputed features.
func (r *Regression) Predict(vars []float64) (float64, error) {
	if !r.Initialised {
		return 0, ErrNotEnoughData
	}

	// apply any features crosses to vars
	for _, cross := range r.Crosses {
		vars = append(vars, cross.Calculate(vars)...)
	}

	p := r.Coeff(0)
	for j := 1; j < len(r.Data[0].Variables)+1; j++ {
		p += r.Coeff(j) * vars[j-1]
	}
	return p, nil
}

// SetObserved sets the name of the observed value.
func (r *Regression) SetObserved(name string) {
	r.Names.Obs = name
}

// GetObserved Gets the name of the observed value.
func (r *Regression) GetObserved() string {
	return r.Names.Obs
}

// SetVar sets the name of variable i.
func (r *Regression) SetVar(i int, name string) {
	if len(r.Names.Vars) == 0 {
		r.Names.Vars = make(map[int]string, 5)
	}
	r.Names.Vars[i] = name
}

// GetVar gets the name of variable i.
func (r *Regression) GetVar(i int) string {
	x := r.Names.Vars[i]
	if x == "" {
		s := []string{"X", strconv.Itoa(i)}
		return strings.Join(s, "")
	}
	return x
}

// AddCross registers a feature cross to be applied to the data points.
func (r *Regression) AddCross(cross featureCross) {
	r.Crosses = append(r.Crosses, cross)
}

// Train the regression with some data points.
func (r *Regression) Train(d ...*DataPoint) {
	r.Data = append(r.Data, d...)
	if len(r.Data) > 2 {
		r.Initialised = true
	}
	return
}

// Apply any feature crosses, generating new observations and updating the data points, as well as
// populating variable names for the feature crosses.
// this should only be run once, as part of Run().
func (r *Regression) applyCrosses() {
	unusedVariableIndexCursor := len(r.Data[0].Variables)
	for _, point := range r.Data {
		for _, cross := range r.Crosses {
			point.Variables = append(point.Variables, cross.Calculate(point.Variables)...)
		}
	}

	if len(r.Names.Vars) == 0 {
		r.Names.Vars = make(map[int]string, 5)
	}
	for _, cross := range r.Crosses {
		unusedVariableIndexCursor += cross.ExtendNames(r.Names.Vars, unusedVariableIndexCursor)
	}
}

// Run determines if there is enough data present to run the regression
// and whether or not the training has already been completed.
// Once the above checks have passed feature crosses are applied if any
// and the model is trained using QR decomposition.
func (r *Regression) Run(logOutput bool) error {
	if !r.Initialised {
		return ErrNotEnoughData
	}

	if r.HasRun {
		return ErrRegressionRun
	}

	//apply any features crosses
	r.applyCrosses()
	r.HasRun = true

	observations := len(r.Data)
	numOfvars := len(r.Data[0].Variables)

	if observations < (numOfvars + 1) {
		return ErrTooManyVars
	}

	// Create some blank variable space
	observed := matrix.Zeros(observations, 1)
	variables := matrix.Zeros(observations, numOfvars+1)

	for i := 0; i < observations; i++ {
		observed.Set(i, 0, r.Data[i].Observed)
		for j := 0; j < numOfvars+1; j++ {
			if j == 0 {
				variables.Set(i, 0, 1)
			} else {
				variables.Set(i, j, r.Data[i].Variables[j-1])
			}
		}
	}

	// Now run the regression
	n := variables.Cols()
	q, reg := variables.QR()
	qty, err := q.Transpose().Times(observed)
	if err != nil {
		return err
	}
	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = qty.Get(i, 0)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * reg.Get(i, j)
		}
		c[i] /= reg.Get(i, i)
	}

	// Output the regression results
	r.CoeffMap = make(map[int]float64, numOfvars)
	for i, val := range c {
		r.CoeffMap[i] = val
		if logOutput {
			if i == 0 {
				r.Formula = fmt.Sprintf("Predicted = %.4f", val)
			} else {
				r.Formula += fmt.Sprintf(" + %v*%.4f", r.GetVar(i-1), val)
			}
		}
	}

	r.calcPredicted(logOutput)
	r.calcVariance()
	r.calcR2()
	r.HasRun = true
	return nil
}

// Coeff returns the calulated coefficient for variable i.
func (r *Regression) Coeff(i int) float64 {
	if len(r.CoeffMap) == 0 {
		return 0
	}
	return r.CoeffMap[i]
}

func (r *Regression) calcPredicted(logOutput bool) string {
	observations := len(r.Data)
	var predicted float64
	var output strings.Builder
	for i := 0; i < observations; i++ {
		r.Data[i].Predicted, _ = r.Predict(r.Data[i].Variables)
		r.Data[i].Error = r.Data[i].Predicted - r.Data[i].Observed
		if logOutput {
			output.WriteString(fmt.Sprintf("%d. observed = %.4f, Predicted = %.4f, Error = %.4f", i, r.Data[i].Observed, predicted, r.Data[i].Error))
		}
	}
	return output.String()
}

func (r *Regression) calcVariance() (int, float64, float64) {
	observations := len(r.Data)
	var obtotal, prtotal, obvar, prvar float64
	for i := 0; i < observations; i++ {
		obtotal += r.Data[i].Observed
		prtotal += r.Data[i].Predicted
	}
	obaverage := obtotal / float64(observations)
	praverage := prtotal / float64(observations)

	for i := 0; i < observations; i++ {
		obvar += math.Pow(r.Data[i].Observed-obaverage, 2)
		prvar += math.Pow(r.Data[i].Predicted-praverage, 2)
	}
	r.Varianceobserved = obvar / float64(observations)
	r.VariancePredicted = prvar / float64(observations)
	return observations, r.Varianceobserved, r.VariancePredicted
}

func (r *Regression) calcR2() {
	r.R2 = r.VariancePredicted / r.Varianceobserved
	return
}

func (r *Regression) printResiduals() string {
	var str strings.Builder
	str.WriteString(fmt.Sprintf("Residuals:\nobserved|\tPredicted|\tResidual\n"))
	for _, d := range r.Data {
		str.WriteString(fmt.Sprintf("%.4f|\t%.4f|\t%.4f\n", d.Observed, d.Predicted, d.Observed-d.Predicted))
	}
	str.WriteString("\n")
	return str.String()
}

// String satisfies the stringer interface to display a dataPoint as a string.
func (d *DataPoint) String() string {
	var str strings.Builder
	str.WriteString(fmt.Sprintf("%.4f", d.Observed))
	for _, v := range d.Variables {
		str.WriteString(fmt.Sprintf("|\t%.2f", v))
	}
	return str.String()
}

// String satisfies the stringer interface to display a regression as a string.
func (r *Regression) String() string {
	if !r.Initialised {
		return ErrNotEnoughData.Error()
	}
	var str strings.Builder
	str.WriteString(r.GetObserved())
	for i := 0; i < len(r.Names.Vars); i++ {
		str.WriteString(fmt.Sprintf("|\t%s", r.GetVar(i)))
	}

	str.WriteString("\n")
	for _, d := range r.Data {
		str.WriteString(fmt.Sprintf("%v\n", d))
	}
	fmt.Println(r.printResiduals())
	str.WriteString(fmt.Sprintf("\nN = %d\nVariance observed = %.4f\nVariance Predicted = %.4f", len(r.Data), r.Varianceobserved, r.VariancePredicted))
	str.WriteString(fmt.Sprintf("\nR2 = %.4f\n", r.R2))
	return str.String()
}

// MakeDataPoints makes a `[]*dataPoint` from a `[][]float64`. The expected fomat for the input is a row-major [][]float64.
// That is to say the first slice represents a row, and the second represents the cols.
// Furthermore it is expected that all the col slices are of the same length.
// The obsIndex parameter indicates which column should be used
func MakeDataPoints(a [][]float64, obsIndex int) []*DataPoint {
	if obsIndex != 0 && obsIndex != len(a[0])-1 {
		return perverseMakeDataPoints(a, obsIndex)
	}

	retVal := make([]*DataPoint, 0, len(a))
	if obsIndex == 0 {
		for _, r := range a {
			retVal = append(retVal, DataPointWrapper(r[0], r[1:]))
		}
		return retVal
	}

	// otherwise the observation is expected to be the last col
	last := len(a[0]) - 1
	for _, r := range a {
		retVal = append(retVal, DataPointWrapper(r[last], r[:last]))
	}
	return retVal
}

func perverseMakeDataPoints(a [][]float64, obsIndex int) []*DataPoint {
	retVal := make([]*DataPoint, 0, len(a))
	for _, r := range a {
		obs := r[obsIndex]
		others := make([]float64, 0, len(r)-1)
		for i, c := range r {
			if i == obsIndex {
				continue
			}
			others = append(others, c)
		}
		retVal = append(retVal, DataPointWrapper(obs, others))
	}
	return retVal
}

// Save json serializes the model
func (r *Regression) Save() ([]byte, error) {
	return json.Marshal(r)
}

// Load places bytes into model
func (r *Regression) Load(data []byte) error {
	return json.Unmarshal(data, r)
}

// SetThreshold ...
func (r *Regression) SetThreshold(thres time.Duration) {
	r.Threshold = thres
}
