# uchit

Uchit @ Spark+AI Summit Europe 2019 -
[Auto-Pilot for Apache Spark Using Machine Learning](https://databricks.com/session_eu19/auto-pilot-for-apache-spark-using-machine-learning)

Uchit helps in auto tuning the configurations for a Spark Application using Machine Learning and other mathematical abstract models.
Auto tuning of Spark configuration is largely a manual effort, needs Big Data domain specific expertise and is not scalable. Uchit would help in automating this process.

## Architecture -

Uchit takes the data from the previous runs of the same application (config values and run time of the application) as input. ML model trained on this data then predicts the best performing config out of the sampled configs.

<p align="center">
    <img width="100" src="https://github.com/qubole/uchit/raw/img/UchitArch.png"/> 
</p>


## Workflow -

1. Data from the previous runs of the same application is used as a training data after being normalized.
2. ML models is trained on the samples gathered in step 1.
3. Using Latin HyperCube sampling(LHS), representative samples are picked from the combinerd domain of the configs so that the entire sample space is covered.
4. Math + any other model, prunes the non-optimal samples using domain specific knowledge.
5. Out of selected samples in step 4, ML model predicts the best performing sample. 
6. Predicted best performing config is denormalized and returned to the user. 

<p align="center"> 
    <img width="100" src="https://github.com/qubole/uchit/raw/img/UchitCombiner.png"/>
</p>
 
 
## How to use - 

### Step 1: Initialize Combiner.

```python
from spark.combiner.combiner import Combiner
combiner = Combiner(4, 26544)
```

- **4** is the **number of cores** per worker node.
- **26544** is the **total memory** per worker node in MB.

### Step 2: Add Training Data

```python
training_data_1 = {
        "spark.executor.memory": 11945,
        "spark.sql.shuffle.partitions": 200,
        "spark.executor.cores": 2,
        "spark.driver.memory": 1024 * 2,
        "spark.sql.autoBroadcastJoinThreshold": 10,
        "spark.sql.statistics.fallBackToHdfs": 0
    }
runtime_in_sec = 248
combiner.add_training_data(training_data_1, runtime_in_sec)
```

### Step 3: Compute best config for next run
```python
best_config = combiner.get_best_config()
print best_config

{'spark.executor.cores': 4, 'spark.driver.memory': 2048, 'spark.sql.statistics.fallBackToHdfs': 1, 'spark.sql.autoBroadcastJoinThreshold': 100, 'spark.executor.memory': 23889, 'spark.sql.shuffle.partitions': 200}
```
 - Compute the best config for next run. 
 - Run the job with the suggested configuration.

### Step 4: Add new training data from Step 3 and repeat the process `n` times

* Run with suggested configs in Step 3 and based upon this run add new training data to model.
* Get the new best config to run the job again.
* Repeat this process for predefined `n` number of times.

#### Please see [Uchit Tutorial](https://github.com/user/repo/blob/branch/other_file.md) for more details.



----
If you have any feedback, feel free to shoot an email to 
[amoghm@qubole.com](mailto:amoghm@qubole.com) or [mayurb@qubole.com](mailto:mayurb@qubole.com)  
