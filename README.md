# Fairness dynamic predictive policing

## Description
Repository containing the code used to simulate dynamic (and not) crime predictions modeled as an allocation problem, and look at the impact of dynamicity in terms of fairness.

## Installation
Install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage
Run the simulation using the following command:

```
python main.py <city>
```

where ```<city>``` must be either ```philadelphia``` or ```los_angeles```, depending on the crime data to run it with.

<br/>

After the simulation ends, the dataframes containing the produced data will be stored in the folder **outputs/[city]**.
These can then be used to create the plots by running the notebook **outputs/plot.ipynb**, which will also store the created plots in the folder **images/[city]**.

## Authors 
Edoardo Merli, edoardo.merli@studio.unibo.it
