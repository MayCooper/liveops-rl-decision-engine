# Economy Policy

The economy policy prevents the RL decision agent from overusing resource grants or power boosts.

Resource grants are blocked when recent reward exposure is high. Temporary power boosts are also blocked under high reward saturation because they function as value-bearing interventions. The benchmark reports average intervention cost and policy violations so that improved win rate is not accepted if it comes from excessive reward spending.
