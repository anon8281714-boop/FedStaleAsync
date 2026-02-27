# AsyncFlower

AsyncFlower is an asynchronous extension of the framework Flower, providing a modular infrastructure to design, configure, and benchmark asynchronous federated learning strategies. 

## **Setup and Run Experiments**

Recommended Python version: 3.12.8

### **Installing dependencies**
```bash
asyncflower $ python -m source venv .venv
asyncflower $ source .venv/bin/activate
(.venv) asyncflower $ pip install -e .
```

### **Running experiments**

(Optional) Set experiments directory (default: `./results`)
```bash
(.venv) $ export EXPERIMENTS_BASE_DIR="<path-to-your-results-dir>"
```

```bash 
(.venv) asyncflower $ python3 run.py --multirun <your-experiment-config-overrides>
```
Note: This command may not work in some shells, such as `zsh`. It was successfully tested on `bash`.
Note: Refer to the `Experiment Configuration` section to learn how to declare configuration variables to customize your experiments. 

## **Experiment Configuration**

This section is under construction...

## **Extending the Implementation**


### **New Methods**

To implement new asynchronous/synchronous federated learning methods, you need to:

1. Implement a new strategy in the `./src/asyncflower/strategy` directory. This strategy must be a subclass of `AsyncStrategy` (see `./src/asyncflower/strategy/async_strategy.py`) and implement all the methods required by that abstract class, noting that it includes all the methods specified in the standard `Strategy` Flower implementation, with a slightly difference in the arguments required by the `aggregate_fit` method, and the additional `round_delimiter` method required in this asynchronous extension. 

2. Create a new `.yaml` config file in the `./config/strategy` directory containing all the required configuration variables to run the method. 

3. Create a builder for your new strategy in the `./src/asyncflower/builders/strategy` directory and add this builder in the `./src/asyncflower/builders/strategy/builder.py` script. Strategy builders receive the configuration variables declared by the user and build an `AsyncStrategy` object. Configurations are loaded in the main file `./run.py`.

### **New Datasets**

This section is under construction...

### **New Models**

This section is under construction...

## **Contribute With This Project**

1. **Flower version:** This implementation uses Flower v1.17.0. Latest versions of Flower provide novel interesting features that can facilitate the implementation of asynchronous methods. It would be interesting to migrate this code to use more recent versions of Flower. 

2. **Find and correct bugs:** If you find any bugs, please report them to us or help us by fixing them. 

3. **New asynchronous methods:** You can implement new asynchronous methods (see the `Implementing New Methods` section).

4. **Documentation and refactoring:** There is a lot of work to do in order to improve the quality of this repository. You can help us by refactoring the code and creating a proper documentation.

5. **Integrate with Flower:** This project does not need to be a separate branch of Flower. It would be interesting if we could merge them so that Flower can offer support to asynchronous federated learning methods. 

## **Contact**

If you have any questions, please contact us: anon8281714@gmail.com.
