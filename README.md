# PO-SMC
 There are 5 main files. Use main.py for test run others for benchmarking purposes.
 There are several maps in the following directories
 ```
pomdpy/config
```
The maps are in txt format. In order to use a different map you need to modify this file 
```
pomdpy/config/robot_pickup_config.json
```
You can create your own map by following this rules.
At first line, define the map size with space for instance 4x13 grid cells: 4 13. 
Use dot for free cells and then use those symbols
* S is the staring position 
* X obstacles 
* G is goal 
* C is a soft constraint (penalty zones) 

For tuning the hyper parameters of PO-SMC algorithm, you need to modify this file 
```
exp_param.yml
```
However, don't choose the minimum number of particles too less, you need to change this file as well
```
variable : minimum_particles = 250 in
pomdpy/discrete_pomdp/discrete_action_mapping.py
```
*minimum_particles* is used to compute the probability of satisfaction of property. 

Finally to disable Statistical Model Checking (SMC), change the SMC flag to False in the following file 

```
pomdpy/action_selection/action_selectors.py
``` 

You can also change the probability theshold *-safety_threshold,* according to your application.

## Hack

If you use PyCharm editor create the STEP keyword in Preferences>Editor>TODO 

```
\bSTEP\b.*
```
and choose any color. It will highlight my line comments in the code.
