# <div align="center">Self-Driving-Car (simulator version)

## Table of Contents

- [About](#about)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
    - [`tools.py`](#toolspy)

## About

This project demonstrates a basic self-driving car model using the Udacity Car Driving Simulator. The goal is to train a Convolutional Neural Network (CNN) to predict the steering angle, allowing the virtual car to drive autonomously at a constant speed. The ultimate objective is for the car to complete a full lap without deviating from the track.

1. Data Collection

The simulator is set to training mode, where we manually drive the car on the track.

During this session, the simulator records essential vehicle sensor data:

Camera images from the virtual car.

Steering angles corresponding to driver inputs.

Speed values to maintain consistency.

2.  Model Training

The recorded data is fed into a CNN model for training.

The model learns to associate different road scenarios with the correct steering angles, such as:

Straight roads

Left and right curves

Lane departures and corrections

3. Autonomous Driving

After training, the model receives real-time images from the simulator’s virtual camera.

It predicts the necessary steering angle and autonomously controls the car.

The goal is to maintain smooth and stable driving throughout the track.

Udacity self driving car simulator is used testing and training our model.

## Structure

```structure
├── environments.yml
├── model
├── notebook
├── term1-simulator-linux
│   └── beta_simulator_Data
│   └── beta_simulator.x86
│   └── beta_simulator.x86_64
├── README.md
├── tools
```

These source directories are as follows:

**Executable Files in `tools`:**

- **`run.py`** - Run the pre-build simulator executable.

**Source Directories:**

- **notebook** - Includes essential files for data visualization and model training.
- **docs** - Includes documents related to algorithms implemented in this project.
- **models** - Includes DL models. 
- **tools** - Includes all the above executable files.
- **term1-simulator-linux** - Contains the program for simulating autonomous vehicles.

## Getting Started

### Installation

1. Clone project repo

    ```
    git clone https://github.com/phatnguyen1510/self-driving_car_using_CNN.git
    ```

2. Set up a virtual environment
    
    This project requires Anaconda for execution. If you do not have Anaconda installed, please download it from [here](https://docs.anaconda.com/anaconda/install/)
    
    Create a virtual environment

    ```
    conda env create -f environments.yml
    ```
    Activate the virtual environment
    ```
    conda activate self-driving-car
    ```
 
### Usage

#### `tools.py`

Run the following command to simulate the self-driving car in autonomous mode

```
python .\tools\run.py
```
