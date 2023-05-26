# online-opt-oversupply
SBU Honors Thesis Project using RHC algorithm to create an online optimization framework for resolving oversupply on the power grid via redistribution of energy to BTC mining and/or battery storage. 

## Mathematical Formulation
Here is the mathematical formulation for the optimization problem, if interested: 

### Variables 


| Variables      | Description |
| ----------- | ----------- |
| **CAISO Variables**     |
| $$t$$   | Current time instance     |
| $$T$$   | Total timeframe     |
| $$P(t)$$   | Curtailed power in CAISO at time instant t      |
| $$p_e(t)$$   | Average (buying) price of electricty on CAISO market|
| **BTC Variables**     |
| $$D(t)$$      | BTC Difficulty at time instant t       |
| $$H(t)$$   | BTC Hashrate at time instant t      |
| $$R$$   | BTC Market Reward|
| $$p_{BTC}(t)$$   | Market price (reward) of BTC      |
| $$c_{BTC}$$   | BTC constant     |
| $$x^P_m(t)$$   | Amount of curtailed power going to BTC mining at time t|
| $$f_m(\boldsymbol{x}^P_m, \boldsymbol{x}^B_m)$$   | Profit from BTC mining      |
| **Battery Variables**     |
| $$B(t)$$   | Charge of battery system at time t |
| $$B_c$$   | Battery Capacity |
| $$B^+$$   | Battery Charging Rate |
| $$B^-$$   | Battery Discharging Rate |
| $$p_s(t)$$   | Average (selling) price of electrcity on CAISO market |
| $$x^P_b(t)$$   | Amount of curtailed power going to battery storage at time t|
| $$f_b(\boldsymbol{x}^P_b, \boldsymbol{x}^B_s)$$   | Profit from battery storage      |
| $$x^B_m(t)$$   | Amount of battery power going to BTC mining at time t|
| $$x^B_s(t)$$   | Amount of battery power sold at time t|


### BTC Mining

$$ c_{BTC} = \frac{R*H(t)}{2^{32}*D(t)}  $$

$$f_{m}(t) = (x^P_m(t) + x^B_m(t)) * p_{BTC}(t) * c_{BTC}(t) - (x^P_m(t) * p_e(t))$$


### Battery Storage

$$B(t) = B(t-1) + x^P_b(t) - x^B_m(t) - x^B_s(t)$$

### Constraints

$$x^P_m(t) \geq 0, \forall t \in [T]$$

$$x^P_b(t) \geq 0, \forall t \in [T]$$

$$x^B_m(t) \geq 0, \forall t \in [T]$$

$$x^B_s(t) \geq 0, \forall t \in [T]$$

$$x^P_m(t) + x^P_b(t)\leq P(t)$$

$$B(t) \leq B_c$$

$$x^P_b(t) \leq B^+$$

$$x^B_m(t) + x^B_s(t) \leq min(B^-, B(t-1))$$

$$min(x^P_b(t), x^B_m(t) + x^B_s(t)) = 0$$


### Objective Function
$$ f_b(\boldsymbol{x}^P_b, \boldsymbol{x}^B_s) = (\boldsymbol{x}^B_s * p_s) - (\boldsymbol{x}^P_b * p_e) $$

$$ f_m(\boldsymbol{x}^P_m, \boldsymbol{x}^B_m) = ((\boldsymbol{x}^P_m + \boldsymbol{x}^B_m) * p_{BTC}(t) * c_{BTC}(t)) - (x^P_m(t) * p_e(t)) $$

$$ max(f_b(\boldsymbol{x}^P_b, \boldsymbol{x}^B_s) + f_m(\boldsymbol{x}^P_m, \boldsymbol{x}^B_m)) $$
