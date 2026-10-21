# Presentation guide

About 8–10 minutes, presented from the website (`docs/index.html` or the GitHub Pages link). Keep the notebook open in another tab to show code if asked.

## Talk flow

| Part | On screen | What to say |
|---|---|---|
| **Opening** (1 min) | Top of the page, stats strip | 34 members, 78 friendships, recorded by Zachary before the club split in two. Our question: can network metrics alone find the key people and predict the split? |
| **1. Metrics** (3 min) | *Metrics* section. Click through Degree → Closeness → Betweenness → Eigenvector → PageRank | For each: read the question it answers, point at the formula, then say who wins. Click member **2** and member **8** to show a member who ranks very differently across metrics. |
| **2. Compare** (1 min) | *Compare* section | All metrics agree on 0 and 33. Officer = popularity/influence, Mr. Hi = reach/brokerage. Degree ≈ PageRank (0.98); eigenvector is the odd one out. |
| **3. Direction** (30 s) | *Direction* section | Karate graph is undirected, so in/out-degree are shown here. Ravi: popular, not active. Eshan: active, not popular. |
| **4. Factions** (2 min) | *Factions* section. Click Real split → Girvan–Newman → Louvain → Nearer leader | Girvan–Newman (betweenness-based) gets 94%. Louvain gets the best modularity but finds 4 groups. "Nearer leader" gets 97%; only member 8 is wrong. 86% of friendships stay inside one side. |
| **5. Robustness** (1 min) | *Robustness* section. Drag the slider 0 → 4 with Betweenness | Removing 4 central members leaves 29% connected vs 86% for random removal. Switch to Degree, slider at 1: removing the Officer first barely matters (97%), removing Mr. Hi first drops to 79%. |
| **Conclusions** (1 min) | *Conclusions* section | Four findings, and the limitation: member 8 changed sides for a personal reason (his black-belt exam), which no structural metric can see. |

## Likely viva questions

**Why is the network undirected? How do you show in/out-degree?**
Friendship ties were recorded as mutual, so there is no direction. In/out-degree is demonstrated on a separate small directed network.

**Why do degree and betweenness pick different winners?**
Degree counts friends; the Officer has 17, Mr. Hi 16. Betweenness counts how often a member lies on shortest paths between others. Many of Mr. Hi's friends connect to the rest of the club only through him: removing him cuts off member 11 and the group {4, 5, 6, 10, 16}, so every path from those 6 members runs through him.

**Why are closeness values so close together (0.28–0.57)?**
The network is small-world: any two members are on average 2.4 steps apart, so nobody is far from anybody.

**How does eigenvector centrality differ from degree?**
It weights each friend by their own importance. Member 2 has only 10 friends but ranks 3rd because those friends include both leaders.

**Why is PageRank almost the same as degree here?**
In an undirected graph, each member's PageRank is spread evenly to their neighbours, so a member's score is driven mostly by how many neighbours they have. The rank correlation is 0.98.

**What is modularity? Why does Louvain beat the real split?**
Modularity measures how much denser ties are inside groups than expected by chance. Louvain maximises it directly and finds 4 tighter sub-groups (Q = 0.43). The real split has Q = 0.39, because people's choices are not purely structural.

**What are ARI and NMI?**
Agreement scores between two groupings, adjusted for chance (1 = identical, 0 = random). They penalise Louvain for returning 4 groups when the truth has 2.

**Why does the repository say 135 triangles but you get 45?**
The repository counts each triangle once at each of its 3 corners: 45 × 3 = 135.

**Limitations?**
Small network (34 nodes), a single snapshot in time, unweighted centrality (edge weights exist and are shown in the data), and structure cannot capture personal motives.
