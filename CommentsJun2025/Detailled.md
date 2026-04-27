Page 9. "Note that a customer can influence only their immediate neighbors, but not the neighbors of those neighbors." -> Why is this?
DK: This is just a simplification for the sake of the model. In reality, influence can propagate through multiple layers of connections, but this would complicate the model significantly. --H.B

P9 "Sets C: Set of campaigns" -> Indicate the domain of this set.
DK: The set C consists of all campaigns that can be executed, where each campaign is defined by its unique identifier. Büyüklüğünü göster.

p9. "rcp priority" -> What type of value can this have? Real or integer?
DK: Integer values 0-100, where 0 is the lowest priority and 100 is the highest.

p9. "b communication limit per customer for the planning horizon (one" -> The possible value here is not clear.
DK: The communication limit is a predefined integer value that represents the maximum number of messages a customer can receive during the planning horizon, typically set based on business rules or customer engagement strategies.

Business domain me

p10. "∀c ∈ C, ∀u ∈ U, . (15)" -> There is an error in the comma.
DK: typo will be fixed.

p10. "Xcuhd ≤ ecu" -> Why can a customer not receive more than 1 message using multichannel?
DK: The model assumes that each customer can only receive one message per campaign through any given channel on a specific day. This is to avoid overwhelming customers with multiple messages, which could lead to disengagement.

p11. "Constraints (8) and (9) limit the number of messages sent to a customer considering these campaign categories; constraints (8) limit the number of communications for campaign categories ignoring the rolling horizon and constraints (9) limit the number of communications for campaign categories for each day." -> Redundant and unclear.
DK: constraint 8 runs over the entire planning horizon, while constraint 9 runs over each day of the planning horizon. This distinction is important to ensure that the model respects both long-term and short-term communication limits.

p12. "We hypothesize that customers who have a history of a CDR and altered their tariff plans within the same short period are likely to be closer contacts and so can influence each other's decisions." -> It's not clear why this is true. 
DK: This hypothesis is based on the assumption that customers who have similar behaviors or experiences (like changing tariff plans) and are already connected an existing social network which is build upon CDR data, are more likely to influence each other's decisions. The underlying idea is that shared experiences can strengthen social ties and increase the likelihood of influence.

p19. "Our HANSA framework" -> Computational complexity?
DK: Shall we add a O(n) analysis for the computational complexity of the model?

p21. Eq 21. On what is it based?
DK: Eq 29 is based on the reinforcement learning mechanism that adapts the neighborhood search strategy based on the performance of different neighborhood operators during the search process. The parameters α and β are empirically determined to balance the importance of improvement count and magnitude in guiding the search.

Learning Components Throughout the search process, the algorithm records performance statistics for each neighborhood operator: NSperf (k) = α·ImprovementCount(k) + β·ImprovementMagnitude(k)
UsageCount(k) (29) where α= 0.7 and β = 0.3 are weighting factors. This reinforcement learning mechanism allows HANSA to focus computational effort on the most productive neighborhood structures for each specific problem instance.

p25. Table 3 -> What is it based on?


p26. "The generated values for these parameters are given in A." -> What is A?  --> APPENDIX

p31. "By strategically targeting influential customers identified through the network structure, the model significantly enhances campaign reach and effectiveness, yielding substantially higher total weighted values compared to approaches that ignore social connectivity during the planning phase."-> Possible "overfitting" of the synthetic network. As discussed before, since the refined social network was built using co-occurrences of tariff changes, the social ties might be overestimated in the model.

p31. Add Discussion section.

p31. Why didn't they at least compare with GNNs for influence maximization in the discussion?

p33. Why were those values in Appendix A chosen?

p33. Publish source code of experiments.


p24. "To implement the methodology detailed in §4.1 we used a real-world social network derived from CDR data, which initially included 114,902 connections among 412,126 individuals." -> But this data is not used in the experiments!! Additionally, where does it come from?