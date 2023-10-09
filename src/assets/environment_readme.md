### soccertwos-og
(Baseline) SoccerTwos with 4 players - trainable blue team and heuristic purple team, tracking win counts via custom side channel, random roles assigned during episode start, og group rewards without role dependency, replicated 1x

### soccertwos-stage1
(CM3 Stage 1) SoccerTwos with 2 players - trainable blue and heuristic purple, tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations, simplified +1/-1 role-based rewards, replicated 1x

### soccertwos-stage2
(CM3 Stage 2) SoccerTwos with 4 players - trainable blue team and heuristic purple team, tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations, simplified +1/-1 role-based rewards, replicated 1x

### soccertwos-stage3
(CM3 Stage 3) SoccerTwos with 4 players - trainable blue team and heuristic purple team, tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations, simplified +1/-1 group rewards WITHOUT role dependency, replicated 1x

### soccertwos-qmix
(Baseline) SoccerTwos with 4 players - trainable blue team and heuristic purple team, tracking win counts via custom side channel, random roles assigned during episode start, simplified +1/-1 group rewards without role dependency, replicated 1x

### soccertwos-mappo-baseline
(Baseline) SoccerTwos with 4 players - trainable blue team and heuristic purple team, tracking win counts via custom side channel, random roles assigned during episode start, simplified +1/-1 group rewards without role dependency, replicated 1x

## Play assets - used for calculating win counts post training
### soccertwos-ppo-heuristic-play-w-roledep
SoccerTwos with 4 players - trainable blue (to be restored with pre-trained policy to be tested) and heuristic purple, tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations, simplified +1/-1 group rewards with role dependency, replicated 1x

### soccertwos-mappo-baseline-heuristic-play-wo-roledep
SoccerTwos with 4 players - trainable blue (to be restored with pre-trained baseline policy to be tested) and heuristic purple, tracking win counts via custom side channel, random roles assigned during episode start, simplified +1/-1 group rewards WITHOUT role dependency, replicated 1x

### soccertwos-ppo-play-w-roledep
SoccerTwos with 4 players - trainable blue and purple (to be restored with pre-trained policy to be tested), tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations, simplified +1/-1 group rewards with role dependency, replicated 1x

### soccertwos-mappo-baseline-play-w-roledep
SoccerTwos with 4 players - trainable blue and purple (to be restored with pre-trained CM3 version to be tested & baseline policy respectively), tracking win counts via custom side channel, random roles assigned during episode start, vectorsensor (i.e. role) added as goal to observations ONLY for blue, simplified +1/-1 group rewards with role dependency, replicated 1x