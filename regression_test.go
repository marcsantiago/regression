package regression

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestRun(t *testing.T) {
	r := new(Regression)
	r.SetObserved("Murders per annum per 1,000,000 inhabitants")
	r.SetVar(0, "Inhabitants")
	r.SetVar(1, "Percent with incomes below $5000")
	r.SetVar(2, "Percent unemployed")
	r.Train(
		DataPointWrapper(11.2, []float64{587000, 16.5, 6.2}),
		DataPointWrapper(13.4, []float64{643000, 20.5, 6.4}),
		DataPointWrapper(40.7, []float64{635000, 26.3, 9.3}),
		DataPointWrapper(5.3, []float64{692000, 16.5, 5.3}),
		DataPointWrapper(24.8, []float64{1248000, 19.2, 7.3}),
		DataPointWrapper(12.7, []float64{643000, 16.5, 5.9}),
		DataPointWrapper(20.9, []float64{1964000, 20.2, 6.4}),
		DataPointWrapper(35.7, []float64{1531000, 21.3, 7.6}),
		DataPointWrapper(8.7, []float64{713000, 17.2, 4.9}),
		DataPointWrapper(9.6, []float64{749000, 14.3, 6.4}),
		DataPointWrapper(14.5, []float64{7895000, 18.1, 6}),
		DataPointWrapper(26.9, []float64{762000, 23.1, 7.4}),
		DataPointWrapper(15.7, []float64{2793000, 19.1, 5.8}),
		DataPointWrapper(36.2, []float64{741000, 24.7, 8.6}),
		DataPointWrapper(18.1, []float64{625000, 18.6, 6.5}),
		DataPointWrapper(28.9, []float64{854000, 24.9, 8.3}),
		DataPointWrapper(14.9, []float64{716000, 17.9, 6.7}),
		DataPointWrapper(25.8, []float64{921000, 22.4, 8.6}),
		DataPointWrapper(21.7, []float64{595000, 20.2, 8.4}),
		DataPointWrapper(25.7, []float64{3353000, 16.9, 6.7}),
	)
	r.Run(false)

	fmt.Printf("Regression formula:\n%v\n", r.Formula)
	fmt.Printf("Regression:\n%s\n", r)

	// All vars are known to positively correlate with the murder rate
	for i, c := range r.CoeffMap {
		if i == 0 {
			// This is the offset and not a coeff
			continue
		}
		if c < 0 {
			t.Errorf("Coefficient is negative, but shouldn't be: %.2f", c)
		}
	}

	//  We know this set has an R^2 above 80
	if r.R2 < 0.8 {
		t.Errorf("R^2 was %.2f, but we expected > 80", r.R2)
	}
}

func TestCrossApply(t *testing.T) {
	r := new(Regression)
	r.SetObserved("Input-Squared plus Input")
	r.SetVar(0, "Input")
	r.Train(
		DataPointWrapper(6, []float64{2}),
		DataPointWrapper(20, []float64{4}),
		DataPointWrapper(30, []float64{5}),
		DataPointWrapper(72, []float64{8}),
		DataPointWrapper(156, []float64{12}),
	)
	r.AddCross(PowCross(0, 2))
	r.AddCross(PowCross(0, 7))
	err := r.Run(false)
	if err != nil {
		t.Error(err)
	}

	fmt.Printf("Regression formula:\n%v\n", r.Formula)
	fmt.Printf("Regression:\n%s\n", r)
	if r.Names.Vars[1] != "(Input)^2" {
		t.Error("Name incorrect")
	}

	for i, c := range r.CoeffMap {
		if i == 0 {
			// This is the offset and not a coeff
			continue
		}
		if c < 0 {
			t.Errorf("Coefficient is negative, but shouldn't be: %.2f", c)
		}
	}

	//  We know this set has an R^2 above 80
	if r.R2 < 0.8 {
		t.Errorf("R^2 was %.2f, but we expected > 80", r.R2)
	}

	// Test that predict uses the cross as well
	val, err := r.Predict([]float64{6})
	if err != nil {
		t.Error(err)
	}
	if val <= 41.999 && val >= 42.001 {
		t.Errorf("Expected 42, got %.2f", val)
	}
}

func TestMakeDataPoints(t *testing.T) {
	a := [][]float64{
		{1, 2, 3, 4},
		{2, 2, 3, 4},
		{3, 2, 3, 4},
	}
	correct := []float64{2, 3, 4}

	dps := MakeDataPoints(a, 0)
	for i, dp := range dps {
		for i, v := range dp.Variables {
			if correct[i] != v {
				t.Errorf("Expected variables to be %v. Got %v instead", correct, dp.Variables)
			}
		}
		if dp.Observed != float64(i+1) {
			t.Error("Expected observed to be the same as the index")
		}
	}

	a = [][]float64{
		{1, 2, 3, 4},
		{1, 2, 3, 4},
		{1, 2, 3, 4},
	}
	correct = []float64{1, 3, 4}
	dps = MakeDataPoints(a, 1)
	for _, dp := range dps {
		for i, v := range dp.Variables {
			if correct[i] != v {
				t.Errorf("Expected variables to be %v. Got %v instead", correct, dp.Variables)
			}
		}
		if dp.Observed != 2.0 {
			t.Error("Expected observed to be the same as the index")
		}
	}

	correct = []float64{1, 2, 3}
	dps = MakeDataPoints(a, 3)
	for _, dp := range dps {
		for i, v := range dp.Variables {
			if correct[i] != v {
				t.Errorf("Expected variables to be %v. Got %v instead", correct, dp.Variables)
			}
		}
		if dp.Observed != 4.0 {
			t.Error("Expected observed to be the same as the index")
		}
	}

}

func TestSaveAndLoad(t *testing.T) {
	r := new(Regression)
	r.SetObserved("Murders per annum per 1,000,000 inhabitants")
	r.SetVar(0, "Inhabitants")
	r.SetVar(1, "Percent with incomes below $5000")
	r.SetVar(2, "Percent unemployed")
	r.Train(
		DataPointWrapper(11.2, []float64{587000, 16.5, 6.2}),
		DataPointWrapper(13.4, []float64{643000, 20.5, 6.4}),
		DataPointWrapper(40.7, []float64{635000, 26.3, 9.3}),
		DataPointWrapper(5.3, []float64{692000, 16.5, 5.3}),
		DataPointWrapper(24.8, []float64{1248000, 19.2, 7.3}),
		DataPointWrapper(12.7, []float64{643000, 16.5, 5.9}),
		DataPointWrapper(20.9, []float64{1964000, 20.2, 6.4}),
		DataPointWrapper(35.7, []float64{1531000, 21.3, 7.6}),
		DataPointWrapper(8.7, []float64{713000, 17.2, 4.9}),
		DataPointWrapper(9.6, []float64{749000, 14.3, 6.4}),
		DataPointWrapper(14.5, []float64{7895000, 18.1, 6}),
		DataPointWrapper(26.9, []float64{762000, 23.1, 7.4}),
		DataPointWrapper(15.7, []float64{2793000, 19.1, 5.8}),
		DataPointWrapper(36.2, []float64{741000, 24.7, 8.6}),
		DataPointWrapper(18.1, []float64{625000, 18.6, 6.5}),
		DataPointWrapper(28.9, []float64{854000, 24.9, 8.3}),
		DataPointWrapper(14.9, []float64{716000, 17.9, 6.7}),
		DataPointWrapper(25.8, []float64{921000, 22.4, 8.6}),
		DataPointWrapper(21.7, []float64{595000, 20.2, 8.4}),
		DataPointWrapper(25.7, []float64{3353000, 16.9, 6.7}),
	)
	r.Run(false)

	savedData, err := r.Save()
	if err != nil {
		t.Fatal(err)
	}

	newR := new(Regression)
	err = newR.Load(savedData)
	if err != nil {
		t.Fatal(err)
	}

	if !cmp.Equal(r, newR) {
		t.Fatalf("Expected r and newR to be the same\nl: %+v\nnewL: %+v", r, newR)
	}

}

func TestNoTrain(t *testing.T) {
	r := new(Regression)
	r.SetObserved("Murders per annum per 1,000,000 inhabitants")
	r.SetVar(0, "Inhabitants")
	r.SetVar(1, "Percent with incomes below $5000")
	r.SetVar(2, "Percent unemployed")
	r.Train(
		DataPointWrapper(11.2, []float64{587000, 16.5, 6.2}),
		DataPointWrapper(13.4, []float64{643000, 20.5, 6.4}),
		DataPointWrapper(40.7, []float64{635000, 26.3, 9.3}),
		DataPointWrapper(5.3, []float64{692000, 16.5, 5.3}),
		DataPointWrapper(24.8, []float64{1248000, 19.2, 7.3}),
		DataPointWrapper(12.7, []float64{643000, 16.5, 5.9}),
		DataPointWrapper(20.9, []float64{1964000, 20.2, 6.4}),
		DataPointWrapper(35.7, []float64{1531000, 21.3, 7.6}),
		DataPointWrapper(8.7, []float64{713000, 17.2, 4.9}),
		DataPointWrapper(9.6, []float64{749000, 14.3, 6.4}),
		DataPointWrapper(14.5, []float64{7895000, 18.1, 6}),
		DataPointWrapper(26.9, []float64{762000, 23.1, 7.4}),
		DataPointWrapper(15.7, []float64{2793000, 19.1, 5.8}),
		DataPointWrapper(36.2, []float64{741000, 24.7, 8.6}),
		DataPointWrapper(18.1, []float64{625000, 18.6, 6.5}),
		DataPointWrapper(28.9, []float64{854000, 24.9, 8.3}),
		DataPointWrapper(14.9, []float64{716000, 17.9, 6.7}),
		DataPointWrapper(25.8, []float64{921000, 22.4, 8.6}),
		DataPointWrapper(21.7, []float64{595000, 20.2, 8.4}),
		DataPointWrapper(25.7, []float64{3353000, 16.9, 6.7}),
	)
	r.SetThreshold(1 * time.Second)
	err := r.Run(false)
	if err != nil {
		t.Error(err)
	}

	oldL := len(r.Data)

	// All vars are known to positively correlate with the murder rate
	for i, c := range r.CoeffMap {
		if i == 0 {
			// This is the offset and not a coeff
			continue
		}
		if c < 0 {
			t.Errorf("Coefficient is negative, but shouldn't be: %.2f", c)
		}
	}

	//  We know this set has an R^2 above 80
	if r.R2 < 0.8 {
		t.Errorf("R^2 was %.2f, but we expected > 80", r.R2)
	}

	time.Sleep(2 * time.Second)
	err = r.Train(DataPointWrapper(25.7, []float64{3353000, 16.9, 6.7}))
	if err != nil {
		t.Errorf("Could not add additional data: %v", err)
	}

	err = r.Run(false)
	if err != nil {
		t.Errorf("Could not run second regression: %v", err)
	}

	// All vars are known to positively correlate with the murder rate
	for i, c := range r.CoeffMap {
		if i == 0 {
			// This is the offset and not a coeff
			continue
		}
		if c < 0 {
			t.Errorf("Coefficient is negative, but shouldn't be: %.2f", c)
		}
	}

	//  We know this set has an R^2 above 80
	if r.R2 < 0.8 {
		t.Errorf("R^2 was %.2f, but we expected > 80", r.R2)
	}

	if oldL+1 != len(r.Data) {
		t.Errorf("one more data point should have been added. r: %d, newR %d", oldL+1, len(r.Data))
	}

}
