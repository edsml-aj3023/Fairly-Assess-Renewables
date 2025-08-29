# Optimization of the UK’s Future Energy Mix Under Diverse Policy Goals

![Generation Mix for Extreme Scenarios](images/newplot.png)

A policy-weighted, multi-zone PyPSA model of the UK grid that designs net-zero pathways by accounting for system level and non-economic costs — comparing tidal’s predictability and wind’s intermittency beyond LCOE.

---

## Project Structure
The project is organized into a modular structure to ensure clarity, reusability, and efficient workflow management. Each component handles a distinct part of the analysis.

**build_network.py (Core Module)**  
Contains the core functions responsible for constructing and parameterizing the PyPSA network, and for extracting the results after optimization. Housing this logic in a separate module promotes clarity and allows for reuse across notebooks. The module includes `assert` statements to validate key model calculations, ensuring reliability.

**model.ipynb (Execution Notebook)**  
The main analysis workflow. It imports functions from `build_network.py` to perform policy sweeps and generate raw output data. This has all the penalties and base costs which can be modifed in case the model is being run at a later date and the costs or penalties have become redundant. 

**plots.ipynb (Analysis & Visualization)**  
Dedicated to post-processing. It transforms raw results from `model.ipynb` into analysis and visualizations. These plots and insights form the basis of the study’s conclusions.

**check_time.ipynb (Utility Notebook)**  
A supplementary tool to estimate runtime and help scope computational experiments by showing how performance scales with the number of time-series snapshots.

---

## Dependencies

To install all dependencies at once, run:

```bash
pip install -r requirements.txt
```
---

## License
This project is licensed under the terms of the [MIT License](LICENSE).