# lighthouse

This is a light-weight, simplified ABM.

## Contents

- `model`: ActivitySim inputs (configs, data) for the lighthouse model. Currently `model/data`
  contains test-scale data from the ActivitySim MTC example model.
- `notebooks`: Demo notebooks to test if the model still works.
- `src/lighthouse`: Python code used to implement this model. Initialially just has a tool to
  download the MTC full-scale example data. But this may grow to include extensions to ActivitySim
  for things we want the lighthouse model to do.
