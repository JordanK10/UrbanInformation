# RileyNotes

Naive observations and questions from notes/jordaninfo.tex and the code. Unfamiliar with the literature on this topic so questions are likely to be subjects of previous study. I'm going through the tex first and the code second, some of my questions will likely be answered later in the documents.

- How are census block groups determined? 
  - Is there any biased signal in their construction (e.g. for elections, school districts, taxes, that kind of stuff) that would influence this analysis if left unaccounted?
  - If so, can that be measured or estimated? How?
  - Googled it: the census people are not the gerrymandering people. Rare win for america.

- Regarding learning rates, are all census blocks assumed to have the same capacity for learning?
  - Need to understand precisely how learning rates are computed.
  - My brief foray into reinforcement learning made me very interested in the parameter $\epsilon$ that controls exploration vs exploitation in e.g. simple bandit problems.
  - I'm curious to see if, for example, the previous resource state of a census block influences its *capacity for* or *risk tolerance for* learning.
  - For example, is it lower-ROI for a resource-constrained block to update beliefs (or, for example, to try to front-run anticipated future changes)
    - if you account for marginal value, a bet that registers as "even odds" (in absolute terms) has much steeper downside for a resource-constrained block as opposed to a resource-rich one
    - another angle: How much can you wager on your beliefs in a noisy environment? Surely that varies based on your stack of chips?
  - *After reading whole latex: learning is measured and variable learning is kinda the whole point, maybe state-dependent K would be cool approach in future*

- I like this macroscopic vs microscopic description of work that bridges scales, I'm going to steal that for tree physiology.

- The survivor bias argument is interesting. I think about this a lot in ecosystems. Let's say there's some global maximum amount of calibration possible given some reward signal to maximize. I don't know if survivorship tells us anything about calibration relative to the absolute maximum. But it does tell us something about the relative calibration of market participants, if they're able to coexist.
  - That might be dependent on assumption about the generation of novel participants though. Imagine a system where there is a constant influx of new participants, with a distribution of calibrations that are *largely* poor compared to the persistent participants. You could constantly slough off poor performers in that system and any given snapshot of the environment could contain drastically imbalanced degrees of coordination.
  - If you have a patchy landscape with an imbalanced distribution of competitive (maybe better term would be *gated* or *unlockable*) resource-rich areas, you could also lose high performers in a similar fashion.
  - So for survivorship assumption to hold, we need our census block groups to represent the same participants in each successive snapshot, without hidden immi/emigration, no hidden population collapse and regrowth, and so on. 
  - Is this measurable? Seems like it would be challenging to track movements between block groups but you might be able to do a in/out for each one.
  - *Seems like beyond the capabilities of the data after closer inspection*

- For an "aggregate win", does that imply someone else loses? Is this a zero sum game? 
  - Or are cities/blocks "winning" portions of new growth, and the zero-sum portion of the game exists in opportunity costs, not real costs?
  - *Was right about opportunity cost over zero-sum* 
  
- What is a CBSA? Census Block seems obvious. SA = Statistical Area? 
  - Google reveals "Core based statistical area"
  - Oh I see. Census tract has to stay within county and state lines. CBSA is more of a city geometry thing and can cross borders.

- Why does win imply a predictable environment? What if e.g. it's easy to win, agents are guessing with poor calibration, and have a 
  - *It makes more sense after seeing the math part*

- $P_t$, predictability, is a "longitudinal frequentist calculation".
  - What is $y_t$? Income change rate maybe? 
    - Oh duh down here it says realization of the growth rate $\gamma_t$
  - I don't understand why uniformity of productivity = predictability
  - Predictability = stuff that's priced in?
    - If something is predictable, doesn't that mean you don't need to learn in order to know about it?
    - Is that meaningful in the context of the analysis done here or is it just a semantic point?
  - From my vantage:
    - Learning = $\Delta$ internal information given some external noisy signal
    - To what extent does $\Delta$ information result in $\Delta$ decision?
    - To what extent does $\Delta$ decision result in $\Delta$ outcome?
    - Re: people that live within block groups, what proportion of economic activity occurs within the block group vs somewhere else?

- Ok the equations are not that complicated after staring at them for a bit.

- The mixture likelihood thing with the $\gamma$ equation:
  - calculate likelihood in win and loss scenarios
  - mix em together using the observed $p_t$ in the whole CBSA

- Does the choice of mostly big cities give us a weird or unrepresentative input dataset where we're more likely to see economic growth than shrink? Does this theory hold closer to real data on the win side than on the loss side? What would happen if you e.g. included a very large number of small, declining cities such that the sample was held close to the avg US economic growth?

- Model assumes that in any given year, the underlying probability of a positive outcome is the same for every single block group within that CBSA.
  - But you need some means to disentangle $k$ from $p_t$
  - tradeoff: lose specificity (e.g. blind to stuff like gentrification in 1 neighborhood, new infra, etc), gain inferential power
