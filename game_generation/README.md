# Generating TWC Games

We provide in this directory the code and data that we used to generate the TWC text-based games.
The TWC dataset that defines all the entities in the games is available in the directory ```twc_dataset```.

The script ```twc_make_game.py``` allows generating TWC games with a specific number of objects and rooms.
As an example, the following command creates and plays a random TWC game with 1 room and 3 objects.
See the requirements and how to set up a conda environment in the main [README](https://github.com/IBM/commonsense-rl).

```bash
$ python twc_make_game.py --objects 3 --rooms 1 --play
```
The complete list of options of the script and their default value can be inspected by running:
```bash
$ python twc_make_game.py -h
```

In the following, we review the most important options.

## Difficulty Levels
The high-level goal of each game is always to tidy up a house by finding objects
and putting them in their correct location. As we have seen, the number of objects and rooms in the game
can be specified using the ```--objects``` and ```--rooms``` options respectively.
Depending on the number of objects and the number of rooms, we have defined 3 difficulty levels:

* ```easy``` level: 1 object and 1 room;
* ```medium``` level: 2/3 objects and 1 room;
* ```hard``` level: 6/7 objects and 1 or 2 rooms.

You can generate games of a specific difficulty level by using the following command:
```bash
$ python twc_make_game.py --level <diff_level>
```
The script also provides an option to generate more than a single game at once.
As an example, the command below generates 5 games
belonging to the ```hard``` difficulty level:
```bash
$ python twc_make_game.py --level hard --num_games 5
```

## Train and Test Distribution

The script provides the possibility of using only a subset of the entities in the dataset to generate the game.
This allows creating games belonging to different distributions that can be used respectively to train and test
the agents.

By default, we reserve 2/3 of the entities for games in the training distribution and 1/3 for games in the test
distribution. You can use the ```--train``` and ```--test``` options respectively to generate a training or a
test game.

As an example, the following command creates a training game in the ```easy``` difficulty level.

```bash
$ python twc_make_game.py --level easy --train
```

A test game of the same difficulty level can be generated as follows:
```bash
$ python twc_make_game.py --level easy --test
```


## Sparse and Dense Rewards
By default, the script generates games with sparse rewards, meaning that only the actions that achieve the goal of
placing a given object in its correct location are rewarded. However, these actions usually have some prerequisite
as the agent first has to find the object within the environment.

In order to generate a game that rewards also some other actions that the agent needs to take in order to achieve
the final goal, you can use the ```--intermediate_reward``` option.

```bash
$ python twc_make_game.py --level easy --intermediate_reward 1
```

The above command generates a game in the ```easy``` difficulty level (1 object and 1 room).
The action that accomplishes the task of finding the object within the room will be rewarded 1 point. 

Rewards must be integer numbers. In case you want actions that place an object in its commonsensical locations to
be rewarded more than the other actions, you can specify the reward as follows:

```bash
$ python twc_make_game.py --level easy --intermediate_reward 1 --reward 2
```

In this case, the task of finding an object will be rewarded 1 point, whereas the task of placing it in its correct
location will be rewarded 2 points.
