## Cleanup Games Generation

See the requirements and how to set up a conda environment in the main [README](../../README.md).

The following command generates and plays a random cleanup game with 1 room (the kitchen) and 5 objects.
 
```bash
$ python iqa_cleanup.py --objects 5 --rooms 1 --initial_room kitchen --play
```

The dataset used to generate the games is available in the directory ```iqa_dataset```.
The complete list of options and their default value can be inspected by running:
```bash
$ python iqa_cleanup.py -h
```

### House Map

The furniture for each room is sampled from the following lists.

##### Kitchen
* fridge
* counter
* dining table
* ladderback chair
* dining chair
* stove
* oven
* kitchen cupboard
* dishwasher
* cutlery drawer
* trash can


##### Pantry
* shelf
* folding chair


##### Livingroom
* sofa
* end table
* coffee table
* side table
* grey carpet
* armchair
* TV stand
* bookcase
* wastepaper basket


##### Bathroom
* toilet
* bathroom cabinet
* pedal bin
* shower
* bathtub
* towel rail
* toilet roll holder
* bath mat
* wall hook
* sink
* dressing table


##### Bedroom
* bed
* nightstand
* wardrobe
* chest of drawers
* desk
* desk chair
* dark carpet
* dressing table


##### Backyard
* clothesline
* patio table
* patio chair
* BBQ
* workbench


##### Corridor
* coat hanger
* hat rack
* umbrella stand
* shoe cabinet
* key holder


##### Laundry room
* washing machine
* laundry basket
* clothes drier
* work table
* bench
* suspended shelf
