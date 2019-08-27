library(readr)
library(dplyr)
library(elo)
library(lubridate)
library(MLmetrics)

# Read in the world cup CSV data
rawdata = read_csv("wc_datathon_dataset.csv")

# Convert the match date to an R date
# rawdata$date = dmy(rawdata$date)

# We're going to use all matches (qualifiers / friendlies etc) AFTER the 2010 world cup to predict the 2014 world cup matches
training = rawdata %>%
    filter(
      # Match should occur after 2010 world cup
      date > rawdata %>% filter(tournament == "World Cup 2010") %>% pull(date) %>% max(), 
      # Match should occur before 2014 world cup starts
      date < rawdata %>% filter(tournament == "World Cup 2014") %>% pull(date) %>% min()
     ) %>% arrange(date)

# We'll be testing our appraoch on the 2014 World Cup
wc_2014 = rawdata %>% filter(tournament == "World Cup 2014")

# Fix the ELO k factor - here you can try different values to see if improves the model performance
k_fac = 20

# Use the elo.run function in this package. 
#  - the score function does the win, loss, draw calculations automatically
elo_run = elo.run(
  score(team_1_goals, team_2_goals) ~ team_1 + team_2,
  data = training,
  k = k_fac
)

# Run predictions on 2014 world cup: the predict function, in this case, just needs the home and away team names for the tournament
wc_2014_home_probabilities = predict(elo_run, newdata = wc_2014 %>% select(team_1, team_2))

## What about draw probailities? ##
# ELO doesn't naturally account for draws. So we'll have to be a little creative to include them in our model.
# Let's look at the historical draw rates for different ELO matchups (proxied by the difference in predicted win probabilities)
#  - For example how often does a 80% - 20% win probability matchup result in a draw
draw_rates = 
  # Create the draw DF from the elo object
  data.frame(
    win_prob = elo_run$elos[,3],
    win_loss_draw = elo_run$elos[,4]
  ) %>%
  # Round the predicted win probabilities to the nearest 0.05
  mutate(
    prob_bucket = 
      abs(
        round(
          (win_prob-(1-win_prob))*20
        )
      ) / 20
  ) %>%
  group_by(prob_bucket) %>%
  # Calculate the rate their was a draw for this win prob - elo package codes a draw as a 0.5
  summarise(
    draw_prob = sum(ifelse(win_loss_draw==0.5, 1, 0)) / n()
  )

# To our WC 2014 dataset let's add in our predicted win probabilities and fold in the expected draw rates from our table above
wc_2014 = wc_2014 %>%
  # Add in probabilities
  mutate(
    home_prob = wc_2014_home_probabilities,
    away_prob = 1 - home_prob,
    # Probability bucket for the draw expectation
    prob_bucket = round(20 * abs((home_prob - away_prob))) / 20
  ) %>%
  # Join in draws
  left_join(draw_rates) %>%
  # Evenly subtract the draw probability from the home and away probality
  mutate(
    home_prob = home_prob - 0.5 * draw_prob,
    away_prob = away_prob - 0.5 * draw_prob
  )

# Our final dataset will keep the teams, predicted probabilities and decode the binary variables for the actual results
#   -  This will be used in our multivariate LogLoss formula which is what this competition will be judged on!
final = wc_2014 %>%
  select(date, team_1, team_2, home_prob, draw_prob, away_prob, team_1_goals, team_2_goals) %>%
  # Binary yes / no for each of the home loss results
  mutate(
    home_actual = ifelse(team_1_goals > team_2_goals, 1, 0),
    draw_actual = ifelse(team_1_goals == team_2_goals, 1, 0),
    away_actual = ifelse(team_2_goals > team_1_goals, 1, 0)
  ) %>%
  select(-team_1_goals, -team_2_goals)


# Calculate the log-loss for this match
MultiLogLoss(
  y_pred = final[,c("home_prob", "draw_prob", "away_prob")],
  y_true = final[,c("home_actual", "draw_actual", "away_actual")]
)

# Logloss: 1.14

# Save Elo Rankings from the elo dataframe
# We can actually just use final.elos()
elo_df <- data.frame(elo_run$elos) %>% mutate(row_ind = seq(1,nrow(elo_run$elos)))

df1 <- elo_df %>% 
  select(X1,X6,row_ind) %>% 
  arrange(row_ind) %>% 
  group_by(X1) %>%
  summarise(current_elo = last(X6),
            row_ind = last(row_ind)) %>% 
  rename(country_ind = X1)

df2 <- elo_df %>% 
  select(X2,X7,row_ind) %>% 
  arrange(row_ind) %>% 
  group_by(X2) %>%
  summarise(current_elo = last(X7),
            row_ind = last(row_ind))%>% 
  rename(country_ind = X2)

elo_rankings <- rbind(df1,df2) %>% ungroup() %>% 
  group_by(country_ind) %>% 
  arrange(desc(row_ind)) %>% 
  filter(row_number()==1)

# Save ELO rankings
country_numbers <- data.frame(country_ind=seq(1,length(elo_run$teams)),country_name=elo_run$teams)
country_elos <- elo_rankings %>% inner_join(country_numbers) %>% select(-row_ind)

write_csv(country_elos,"country_elos.csv")
