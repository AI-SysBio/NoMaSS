# NoMaSS: A Scalable and Scalable Framework for complex Non-Markovian Gillespie-based Stochastic Simulation

<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/REGIR/master/Figures/REGIR.png" width=400>
Discrete stochastic processes are widespread in both nature and human-made systems, with applications across physics, biochemistry, epidemiology, social patterns and finance, just to name a few. In the majority of these systems, the dynamics cannot be properly described with memoryless (or Markovian) interactions, and thus require the use of numerical tools for analyzing these non-Markovian dynamics. This repository contains the implementattion of a general and scalable framework to simulate non-Markovian stochastic systems with arbitrary inter-event time distribution, mixing delay based methods, MOSAIC and Laplace Gillespie [1].

&nbsp;



        
        
### Simulating a non-Markovian system

First, you need to install NoMaSS, or you can use the `NoMaSS.py` file provided in the repository:

	- pip install nomass


Then, you can run a non-Markovian simulation with the toy example below. Other examples, including the three biochemical systems described in the paper: Cell division, differentiation and RNA transcription, are provided in the `/NoMaSS/Examples` and `/REGIR/Biochemical_applications` folders.

	import nomass

	#Set the simulation parameters:
	class param:
		Tend = 10		#Length of the simulation
		N_simulations = 20	#The simulation results should be averaged over many trials
		unit = 'h'		#Unit of time (used for plotting only)
		timepoints = 100	#Number of timepoints to record (used for plotting only)

	r1 = 1
	r2 = 4
	r3 = 0.03
	alpha1 = 20
	alpha2 = 5
      
	#Define the reaction chanels:
	reaction1 = nomass.Reaction_channel(param,rate=r1, shape_param=alpha1, distribution = 'Gamma', method = 'Delay')
	reaction1.reactants = ['A']
	reaction1.products = ['B']	
	reaction2 = nomass.Reaction_channel(param,rate=r2, shape_param=alpha2, distribution = 'Weibull', method = 'MOSAIC')
	reaction2.reactants = ['B']
	reaction2.products = ['C','A']
	reaction3 = nomass.Reaction_channel(param, rate=r3, shape_param=alpha2, distribution = 'Gamma', method = 'Laplace')
	reaction3.reactants = ['C','D']
	reaction3.products = []
	reaction3 = nomass.Reaction_channel(param, rate=r3)
	reaction3.reactants = ['A','B']
	reaction3.products = []
		
	
	#Define the initial population of reactants:
	N_init = {'A':300,'B':0,'C':0,'D':0}

	#Initialize and run the Gillepsie simulation:
	SSA_simul = nomass.Gillespie_simulation(N_init,param)
	SSA_simul.reaction_channel_list = [reaction1, reaction2, reaction3]
	populations = SSA_simul.run_simulations(param.Tend, verbose = True)
	
	#Plot the results:
	SSA_simul.plot_inter_event_time_distribution()
	SSA_simul.plot_populations()

The algorithm runs for a few seconds and output the following figures (note that you can disables all printing by passing the argument `verbose = False` when running the simulation):
<p align="center">
  <img src="https://raw.githubusercontent.com/Aurelien-Pelissier/REGIR/master/Figures/REGIR_test.png" width=800>
</p>

The oscillations resulting from the markovian dynamics are clearly visible. If you check carefully, you will notice that the *theoretical distributions* do not match exactly the *simulated distributions*, even if you increase the number of simulations. This happens because the entities A and B are reactants of two reaction channels at the same time, and the *theoretical distribution* only represent the inter-event time distribution that **the reaction channel would have if it was the only process interaction with that reactant**. In practice, these kind of situations will occur frequently in non-Markovian systems, so do not worry if the simulated and theoretical distributions do not match exactly. The accuracy of REGIR was rigourously demonstrated in [1] (see the `/REGIR/Benchmark` folder).
      
### Implemented distributions

With the current implementation, each available distribution are characterised by their rate and a shape parameter as follow:

| PDF name           | DelaySSA |    MOSAIC   |   Laplace   |
| ------------------ | :------: | :---------: | :---------: |
| Exponential        |     ✅    |      ✅      |      ✅      |
| Gamma              |     ✅    | ✅ *(α ≥ 1)* | ✅ *(α ≤ 1)* |
| Weibull            |     ✅    | ✅ *(α ≥ 1)* | ✅ *(α ≤ 1)* |
| Gaussian / Normal  |     ✅    |      ✅      |      ❌      |
| Lognormal          |     ✅    |      ✅      |      ❌      |
| Cauchy             |     ✅    |      ✅      |      ❌      |
| Power-law (Pareto) |     ✅    |      ✅      |      ❌      |
| Power-law (Lomax)  |     ✅    |      ❌      |      ✅      |

*Supported inter-event time distributions (IEDs) across NoMaSS backend methods. Each distribution is parameterized by a nominal rate λ (inverse mean waiting time — or median when the mean is undefined) and a shape parameter α; constraints on α are indicated where applicable. Full IED parametrizations are provided in Table `tab:Nomass-param`.*


With the current implementation, each available distribution are characterised by their rate and a shape parameter as follow:

      Exponential:
          - rate: 1/mean
          - shape parameter: None
      
      Normal:
          - rate: 1/mean
          - shape: std/mean
      
      LogNormal:
          - rate: 1/mean
          - shape: std/mean
          
      Gamma:
          - rate: 1/mean
          - shape: α >= 1 (https://en.wikipedia.org/wiki/Gamma_distribution)
          
      Weibull:
          - rate: 1/mean
          - shape: k >= 1 (https://en.wikipedia.org/wiki/Weibull_distribution)
          
      Cauchy:
          - rate: 1/median
          - shape: γ (https://en.wikipedia.org/wiki/Cauchy_distribution)
      

Keep in mind that non-Markovian simulations are only available for reaction channels with a single reactant, as the definition of inter-event time distribution is ambigious for channels with multiple reactants. If a channel is defined without or with more than one reactant, it will be considered as a Poisson process. Also, note that monotolically decreasing distributions, such as Weibull (k < 1), gamma (α < 1) or power laws, are not available in the current implementation of this repository, as these can be more elegantly and efficiently simulated with the Laplace Gillespie algorithm (LGA) [2]. 

*Feel free to drop me an email if you have interest in me adding the Laplace Gillespie or any other relevant distributions to this implementation.* 


### Customizing REGIR for your system

The REGIR framework offer countless possibilities and highly customizable models. However, with the current implementation, reactions propensities are always proportional to the number of reactant. For example, the reaction (A+B -> C) will have a propensity of *a = A x B x r*. In some models, you might want to implement more complex formula for the reaction propensities, (such as for example *a = A x B x r / D*, where D is a parameter that evolves with the system). To do so, you can  directly modify the `REGIR/compute_propensities` function according to your need. Likewise, you might want to set up specific rejection rules if your reactants have some individual properties, and modify them appropriatly. To do so, first define your reactant properties in the `REGIR/Reactant` class, and then define your reaction specific rules in `REGIR/perform_reaction`.

*Feel free to email me if you are not sure how to do it, I will be happy to help !*


## References

[1] Pélissier, A, Phan, M, et al. "Gillespie-Based Methods for Non-Markovian Stochastic Simulation: A Practical Guide". *Under Review* (2026)

[2] Pélissier, A, Phan, M, et al. "Unifying non-Markovian Dynamics and Agent Heterogeneity in Scalable Stochastic Networks". *Nature Communication* (2026) [[Preprint]](https://arxiv.org/abs/2212.05059)

[3] Masuda, Naoki, and Luis EC Rocha. "A Gillespie algorithm for non-Markovian stochastic processes." *Siam Review 60.1* (2018): 95-115.

[4] Xiaodong Cai. “Exact stochastic simulation of coupled chemical reactions with delays”. *The Journal of chemical physics 126.12* (2007).

[5] David F Anderson. “A modified next reaction method for simulating chemical systems with time dependent propensities and delays”. *The Journal of chemical physics 127.21* (2007).
