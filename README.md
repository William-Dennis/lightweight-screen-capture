# python-template

rl experimentation with slay the spire game

# game setup

install the game on steam with the following mods
- StSLib https://steamcommunity.com/sharedfiles/filedetails/?id=1609158507
- BaseMod https://steamcommunity.com/sharedfiles/filedetails/?id=1605833019
- ModTheSpire https://steamcommunity.com/sharedfiles/filedetails/?id=1605060445
- Keyboard Shortcuts https://steamcommunity.com/sharedfiles/filedetails/?id=2173245479

# Action Space

*We want a minimal action space so i propose*
Keys
- 1 -> for special commands
- rightarrow -> for iterating
- enter -> for activating cards
- e -> for ending the round

# RL Loop

Screen grab -> determine if it is the players turn -> CNN -> obersvation inputs -> action -> calculate reward

# Reward

- Primary reward will be the stage they are on
- Secondary reward will be player health