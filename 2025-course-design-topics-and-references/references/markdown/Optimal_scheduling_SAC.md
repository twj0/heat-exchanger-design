# Optimal scheduling strategy of electricity and thermal energy storage based on SAC approach

## Abstract

This paper proposes an **optimal scheduling strategy** for electricity and thermal energy storage systems in community microgrids, using the **Soft Actor-Critic (SAC)** reinforcement learning algorithm. The objective is to minimize operational costs while maintaining system stability and thermal comfort, under the uncertainty of renewable generation and load variations.

---

## 1. Introduction

Community microgrids integrate distributed energy resources (DERs) such as photovoltaic (PV) systems, combined heat and power (CHP) units, electric energy storage (EES), and thermal energy storage (TES). Optimal operation requires coordination between electric and thermal subsystems. Traditional optimization methods (e.g., MILP, DP) are limited by computational burden and lack of adaptability to dynamic environments.

To address these challenges, this paper develops a **model-free SAC-based control framework** for real-time scheduling of electricity and thermal storage, capable of learning optimal policies under stochastic conditions.

---

## 2. System Model

### 2.1 Overview of Community Microgrid

The microgrid consists of:
- **Electric load** and **thermal load**
- **CHP unit** producing both electricity and heat
- **Heat pump (HP)** utilizing electricity to provide heating
- **Electric Energy Storage (EES)** and **Thermal Energy Storage (TES)** subsystems

*Figure 1. Structure of the electricity-thermal coupled microgrid system.*

### 2.2 CHP Model

The CHP unit output relationships are given by:

$$
P_{CHP} = \eta_e \cdot F_{CHP}
$$

$$
Q_{CHP} = \eta_h \cdot F_{CHP}
$$

where $\eta_e$ and $\eta_h$ are the electric and thermal efficiencies, respectively, and $F_{CHP}$ denotes the fuel input.

### 2.3 Heat Pump Model

$$
Q_{HP} = COP \cdot P_{HP}
$$

where $COP$ is the coefficient of performance and $P_{HP}$ is the electrical power consumed.

### 2.4 Energy Storage Model

The state of charge (SOC) dynamics for EES and TES are modeled as:

$$
SOC_{t+1}^{EES} = SOC_t^{EES} + \frac{\eta_{ch} P_{ch} - P_{dis}/\eta_{dis}}{C_{EES}}
$$

$$
SOC_{t+1}^{TES} = SOC_t^{TES} + \frac{Q_{ch} - Q_{dis}}{C_{TES}}
$$

---

## 3. Optimization Framework

### 3.1 Markov Decision Process (MDP)

State $s_t$, action $a_t$, and reward $r_t$ are defined as:

$$
r_t = - (C_{elec}(t) + C_{gas}(t) + C_{deg}(t))
$$

where:
- $C_{elec}(t)$: electricity purchase cost
- $C_{gas}(t)$: gas consumption cost
- $C_{deg}(t)$: battery degradation cost

### 3.2 SAC Algorithm Formulation

The objective function with entropy regularization is:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]
$$

where $\alpha$ is the temperature parameter controlling the entropy weight.

The two Q-functions are updated by minimizing:

$$
L_Q(\theta_i) = \mathbb{E}[(Q_{\theta_i}(s_t,a_t) - y_t)^2]
$$

with target value:

$$
y_t = r_t + \gamma (\min_{i=1,2} Q_{\bar{\theta}_i}(s_{t+1},a_{t+1}) - \alpha \log \pi(a_{t+1}|s_{t+1}))
$$

The policy network is trained by minimizing:

$$
L_\pi(\phi) = \mathbb{E}[\alpha \log \pi_\phi(a_t|s_t) - Q_\theta(s_t,a_t)]
$$

### 3.3 Algorithm Flow

*Figure 2. Flowchart of the Soft Actor-Critic algorithm for energy management.*

---

## 4. Simulation and Results

### 4.1 Experimental Setup

Simulation uses data from California ISO (CAISO). Load and renewable profiles correspond to typical summer and winter days.

| Parameter | Value | Unit |
|------------|--------|------|
| Simulation horizon | 24 | h |
| Time step | 1 | h |
| Battery capacity | 200 | kWh |
| TES capacity | 300 | kWh_th |
| CHP efficiency | 0.35 (electric) / 0.45 (thermal) | — |

### 4.2 Convergence Analysis

*Figure 3. Convergence curve of SAC algorithm over episodes (Summer/Winter).*

The cumulative reward stabilizes after ~200 episodes, showing strong learning stability.

### 4.3 Scheduling Results

*Figure 4. Dispatch results over 3 days (Summer scenario).*

- EES charges at low-price periods and discharges during peaks.
- CHP and HP coordinate to satisfy heating demand with minimal cost.

### 4.4 Performance Comparison

| Algorithm | Summer Cost ($) | Winter Cost ($) | Decision Time (s) |
|------------|-----------------|-----------------|------------------:|
| SAC | **806.85** | **978.74** | **7.92** |
| DDPG | 874.23 | 1053.64 | 12.63 |
| PPO | 905.43 | 1102.33 | 15.26 |
| DQN | 1053.43 | 1250.63 | 20.35 |

*Figure 5. Comparison of operating cost under different RL algorithms.*

---

## 5. Discussion

The SAC algorithm demonstrates superior adaptability and convergence compared to conventional reinforcement learning methods. It can effectively manage stochastic variations in renewable generation and demand, ensuring system stability and economic performance.

---

## 6. Conclusion

This study develops a **SAC-based optimal scheduling strategy** for hybrid electricity-thermal energy storage systems in community microgrids. The proposed method achieves significant cost reduction and enhanced computational efficiency.

Future work will focus on **multi-agent reinforcement learning (MARL)** and **distributed control mechanisms** considering communication and privacy constraints.

---

## References

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.* ICML.
2. Zhang, H., Liu, Y., et al. (2024). *Optimal operation of electricity-thermal systems based on deep reinforcement learning.* J. Energy Storage.
3. Li, J., et al. (2023). *Coordinated control of CHP and energy storage for microgrids.* Applied Energy.

---

**End of Document**
