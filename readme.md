#  Risk Assessment of Injuries in NBA Players

```
Boettcher Moritz, Canepa Lucio, Marrucchiello Carmen, Sidler Leva
```

## Data

the raw dataset, downloaded from [kaggle](https://www.kaggle.com/datasets/icliu30/nba-player-stats-and-injured-data-from-13-to-23), is `player_stats_inuries.csv` under `/data`. The `/data` folder contains all generated datasets starting from the original one.
It can be seen in detail how the datasets are generated in the `data_tidying.ipynb` notebook:

- `data/player_stats_extended.csv`: added a severity column
- `data/players.csv`: contains selected propreties for unique players
- `data/seasons.csv`: contains aggregate information filtered by season
- `data/teams_summary.csv`: information over number of players and season by teams
- `data/injuries/{key}.csv`: dataset filtered by injury type
- `data/teams/{key}.csv`: dataset filtered by teams

## Notebooks

- `data_tidying`: responsible to generate all datasets listed above and investigate basic information about quality and coherence of the data.
- `descriptive_analysis`: contains all first generated plots, allowing to gather a first better understanding of the data.
- `clustering`: contains all the code investigating the best way to cluster the data and display the results.
- `regression`: contains all the code responsible to retrieve the data as needed and perform multiple linear and logistic refression.
- `random_forest`: responsible to run the random forest simulation and explanatory visualizations.
- `tools.py`: contains reusable functions and utils imported by some notebooks.
