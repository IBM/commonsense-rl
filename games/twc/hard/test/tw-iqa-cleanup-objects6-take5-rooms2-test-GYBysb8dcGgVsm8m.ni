Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
a thing can be drinkable. a thing is usually not drinkable. a thing can be cookable. a thing is usually not cookable. a thing can be damaged. a thing is usually not damaged. a thing can be sharp. a thing is usually not sharp. a thing can be cuttable. a thing is usually not cuttable. a thing can be a source of heat. Type of cooking is a kind of value. The type of cooking are raw, grilled, roasted and fried. a thing can be needs cooking. Type of cutting is a kind of value. The type of cutting are uncut, sliced, diced and chopped.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
A room has a text called internal name.


The carrying capacity of the player is 0.


The r_0 and the r_1 are rooms.

The internal name of r_0 is "bedroom".
The printed name of r_0 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "You've just shown up in a bedroom. You can barely contain your excitement.

 What's that over there? It looks like it's a wardrobe.[if c_0 is open and there is something in the c_0] The wardrobe contains [a list of things in the c_0]. Now why would someone leave that there?[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is "[if c_0 is open and the c_0 contains nothing] What a letdown! The wardrobe is empty![end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is " [if c_1 is locked]A locked[else if c_1 is open]An open[otherwise]A closed[end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is " chest of drawers is close by.[if c_1 is open and there is something in the c_1] The chest of drawers contains [a list of things in the c_1]. Classic TextWorld.[end if]".
The bedroom part 4 is some text that varies. The bedroom part 4 is "[if c_1 is open and the c_1 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bedroom part 5 is some text that varies. The bedroom part 5 is " You can make out a bed. The bed is large.[if there is something on the s_0] On the bed you make out [a list of things on the s_0]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The bedroom part 6 is some text that varies. The bedroom part 6 is "[if there is nothing on the s_0] But the thing is empty, unfortunately. What, you think everything in TextWorld should have stuff on it?[end if]".
The bedroom part 7 is some text that varies. The bedroom part 7 is " You make out a desk chair. The desk chair is usual.[if there is something on the s_1] On the desk chair you see [a list of things on the s_1]. You shudder, but continue examining the room.[end if]".
The bedroom part 8 is some text that varies. The bedroom part 8 is "[if there is nothing on the s_1] Looks like someone's already been here and taken everything off it, though.[end if]".
The bedroom part 9 is some text that varies. The bedroom part 9 is " You make out a dark carpet. Why don't you take a picture of it, it'll last longer! [if there is something on the s_2]On the dark carpet you make out [a list of things on the s_2].[end if]".
The bedroom part 10 is some text that varies. The bedroom part 10 is "[if there is nothing on the s_2]But the thing hasn't got anything on it.[end if]".
The bedroom part 11 is some text that varies. The bedroom part 11 is " You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a nightstand. I guess it's true what they say, if you're looking for a nightstand, go to TextWorld. [if there is something on the s_3]You see [a list of things on the s_3] on the nightstand, so there's that.[end if]".
The bedroom part 12 is some text that varies. The bedroom part 12 is "[if there is nothing on the s_3]However, the nightstand, like an empty nightstand, has nothing on it.[end if]".
The bedroom part 13 is some text that varies. The bedroom part 13 is " You can make out a dressing table. [if there is something on the s_4]You see [a list of things on the s_4] on the dressing table.[end if]".
The bedroom part 14 is some text that varies. The bedroom part 14 is "[if there is nothing on the s_4]But the thing is empty, unfortunately.[end if]".
The bedroom part 15 is some text that varies. The bedroom part 15 is " Look over there! a desk. The desk is normal.[if there is something on the s_5] On the desk you can make out [a list of things on the s_5].[end if]".
The bedroom part 16 is some text that varies. The bedroom part 16 is "[if there is nothing on the s_5] However, the desk, like an empty desk, has nothing on it. You make a mental note to not get your hopes up the next time you see a desk in a room.[end if]".
The bedroom part 17 is some text that varies. The bedroom part 17 is "

You don't like doors? Why not try going south, that entranceway is not blocked by one.".
The description of r_0 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3][bedroom part 4][bedroom part 5][bedroom part 6][bedroom part 7][bedroom part 8][bedroom part 9][bedroom part 10][bedroom part 11][bedroom part 12][bedroom part 13][bedroom part 14][bedroom part 15][bedroom part 16][bedroom part 17]".

The r_1 is mapped south of r_0.
The internal name of r_1 is "corridor".
The printed name of r_1 is "-= Corridor =-".
The corridor part 0 is some text that varies. The corridor part 0 is "Well, here we are in a corridor.

 You can make out [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The corridor part 1 is some text that varies. The corridor part 1 is " shoe cabinet.[if c_2 is open and there is something in the c_2] The shoe cabinet contains [a list of things in the c_2], so there's that.[end if]".
The corridor part 2 is some text that varies. The corridor part 2 is "[if c_2 is open and the c_2 contains nothing] The shoe cabinet is empty! What a waste of a day![end if]".
The corridor part 3 is some text that varies. The corridor part 3 is " You scan the room for a hat rack, and you find a hat rack. [if there is something on the s_6]You see [a list of things on the s_6] on the hat rack. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The corridor part 4 is some text that varies. The corridor part 4 is "[if there is nothing on the s_6]However, the hat rack, like an empty hat rack, has nothing on it.[end if]".
The corridor part 5 is some text that varies. The corridor part 5 is " You rest your hand against a wall, but you miss the wall and fall onto an umbrella stand. [if there is something on the s_7]On the umbrella stand you make out [a list of things on the s_7].[end if]".
The corridor part 6 is some text that varies. The corridor part 6 is "[if there is nothing on the s_7]But the thing is empty.[end if]".
The corridor part 7 is some text that varies. The corridor part 7 is " You can make out a key holder. [if there is something on the s_8]On the key holder you can make out [a list of things on the s_8].[end if]".
The corridor part 8 is some text that varies. The corridor part 8 is "[if there is nothing on the s_8]But there isn't a thing on it. What, you think everything in TextWorld should have stuff on it?[end if]".
The corridor part 9 is some text that varies. The corridor part 9 is " You see a coat hanger. [if there is something on the s_9]On the coat hanger you can see [a list of things on the s_9].[end if]".
The corridor part 10 is some text that varies. The corridor part 10 is "[if there is nothing on the s_9]But the thing hasn't got anything on it. Hopefully this doesn't make you too upset.[end if]".
The corridor part 11 is some text that varies. The corridor part 11 is "

There is an exit to the north. Don't worry, there is no door.".
The description of r_1 is "[corridor part 0][corridor part 1][corridor part 2][corridor part 3][corridor part 4][corridor part 5][corridor part 6][corridor part 7][corridor part 8][corridor part 9][corridor part 10][corridor part 11]".

The r_0 is mapped north of r_1.

The c_0 and the c_1 and the c_2 are containers.
The c_0 and the c_1 and the c_2 are privately-named.
The o_0 and the o_3 and the o_1 and the o_2 and the o_4 and the o_5 are object-likes.
The o_0 and the o_3 and the o_1 and the o_2 and the o_4 and the o_5 are privately-named.
The r_0 and the r_1 are rooms.
The r_0 and the r_1 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.
The slot_1 and the slot_10 and the slot_11 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 and the slot_0 are things.
The slot_1 and the slot_10 and the slot_11 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 and the slot_0 are privately-named.

The description of c_0 is "The [noun] looks ominous. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to break open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_0 is "wardrobe".
Understand "wardrobe" as c_0.
The c_0 is in r_0.
The c_0 is open.
The description of c_1 is "The [noun] looks towering. [if open]It is open.[else if locked]It is locked.[otherwise]It is closed.[end if]".
The printed name of c_1 is "chest of drawers".
Understand "chest of drawers" as c_1.
Understand "chest" as c_1.
Understand "drawers" as c_1.
The c_1 is in r_0.
The c_1 is open.
The description of c_2 is "The [noun] looks grand. [if open]It is open.[else if locked]It is locked.[otherwise]It is closed.[end if]".
The printed name of c_2 is "shoe cabinet".
Understand "shoe cabinet" as c_2.
Understand "shoe" as c_2.
Understand "cabinet" as c_2.
The c_2 is in r_1.
The c_2 is open.
The description of o_0 is "The [noun] is antiquated.".
The printed name of o_0 is "clean white socks".
The indefinite article of o_0 is "a pair of".
Understand "clean white socks" as o_0.
Understand "clean" as o_0.
Understand "white" as o_0.
Understand "socks" as o_0.
The o_0 is in r_1.
The description of s_0 is "The [noun] is shaky.".
The printed name of s_0 is "bed".
Understand "bed" as s_0.
The s_0 is in r_0.
The description of s_1 is "The [noun] is unstable.".
The printed name of s_1 is "desk chair".
Understand "desk chair" as s_1.
Understand "desk" as s_1.
Understand "chair" as s_1.
The s_1 is in r_0.
The description of s_2 is "The [noun] is unstable.".
The printed name of s_2 is "dark carpet".
Understand "dark carpet" as s_2.
Understand "dark" as s_2.
Understand "carpet" as s_2.
The s_2 is in r_0.
The description of s_3 is "The [noun] is undependable.".
The printed name of s_3 is "nightstand".
Understand "nightstand" as s_3.
The s_3 is in r_0.
The description of s_4 is "The [noun] is unstable.".
The printed name of s_4 is "dressing table".
Understand "dressing table" as s_4.
Understand "dressing" as s_4.
Understand "table" as s_4.
The s_4 is in r_0.
The description of s_5 is "The [noun] is undependable.".
The printed name of s_5 is "desk".
Understand "desk" as s_5.
The s_5 is in r_0.
The description of s_6 is "The [noun] is wobbly.".
The printed name of s_6 is "hat rack".
Understand "hat rack" as s_6.
Understand "hat" as s_6.
Understand "rack" as s_6.
The s_6 is in r_1.
The description of s_7 is "The [noun] is wobbly.".
The printed name of s_7 is "umbrella stand".
Understand "umbrella stand" as s_7.
Understand "umbrella" as s_7.
Understand "stand" as s_7.
The s_7 is in r_1.
The description of s_8 is "The [noun] is an unstable piece of garbage.".
The printed name of s_8 is "key holder".
Understand "key holder" as s_8.
Understand "key" as s_8.
Understand "holder" as s_8.
The s_8 is in r_1.
The description of s_9 is "The [noun] is solidly built.".
The printed name of s_9 is "coat hanger".
Understand "coat hanger" as s_9.
Understand "coat" as s_9.
Understand "hanger" as s_9.
The s_9 is in r_1.
The description of slot_1 is "".
The printed name of slot_1 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_10 is "".
The printed name of slot_10 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_11 is "".
The printed name of slot_11 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_2 is "".
The printed name of slot_2 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_3 is "".
The printed name of slot_3 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_4 is "".
The printed name of slot_4 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_5 is "".
The printed name of slot_5 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_6 is "".
The printed name of slot_6 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_7 is "".
The printed name of slot_7 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_8 is "".
The printed name of slot_8 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_9 is "".
The printed name of slot_9 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of o_3 is "The [noun] would seem to be out of place here".
The printed name of o_3 is "blue moccasins".
The indefinite article of o_3 is "a pair of".
Understand "blue moccasins" as o_3.
Understand "blue" as o_3.
Understand "moccasins" as o_3.
The player carries the o_3.
The description of o_1 is "The [noun] is dirty.".
The printed name of o_1 is "clean azure skirt".
Understand "clean azure skirt" as o_1.
Understand "clean" as o_1.
Understand "azure" as o_1.
Understand "skirt" as o_1.
The o_1 is on the s_4.
The description of o_2 is "The [noun] is cheap looking.".
The printed name of o_2 is "clean white skirt".
Understand "clean white skirt" as o_2.
Understand "clean" as o_2.
Understand "white" as o_2.
Understand "skirt" as o_2.
The o_2 is on the s_9.
The description of o_4 is "The [noun] is modern.".
The printed name of o_4 is "brown moccasins".
The indefinite article of o_4 is "a pair of".
Understand "brown moccasins" as o_4.
Understand "brown" as o_4.
Understand "moccasins" as o_4.
The o_4 is on the s_3.
The description of o_5 is "The [noun] would seem to be to fit in here".
The printed name of o_5 is "clean plaid pullover".
Understand "clean plaid pullover" as o_5.
Understand "clean" as o_5.
Understand "plaid" as o_5.
Understand "pullover" as o_5.
The o_5 is on the s_3.
The description of slot_0 is "".
The printed name of slot_0 is "".
When play begins, increase the carrying capacity of the player by 1..


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with ""

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The o_0 is in the c_1:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with ""

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The o_1 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with ""

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The o_2 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest2 completed is true;

The quest3 completed is a truth state that varies.
The quest3 completed is usually false.

Test quest3_0 with ""

Every turn:
	if quest3 completed is true:
		do nothing;
	else if The o_3 is in the c_2:
		increase the score by 1; [Quest completed]
		Now the quest3 completed is true;

The quest4 completed is a truth state that varies.
The quest4 completed is usually false.

Test quest4_0 with ""

Every turn:
	if quest4 completed is true:
		do nothing;
	else if The o_4 is in the c_2:
		increase the score by 1; [Quest completed]
		Now the quest4 completed is true;

The quest5 completed is a truth state that varies.
The quest5 completed is usually false.

Test quest5_0 with ""

Every turn:
	if quest5 completed is true:
		do nothing;
	else if The o_5 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest5 completed is true;

Use scoring. The maximum score is 6.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		say "Your score has just gone up by [V in words] ";
		if V > 1:
			say "points.";
		else:
			say "point.";
		Now the last notified score is the score;
	if score is maximum score:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	say "You scored [score] out of a possible [maximum score], in [turn count] turn(s).";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "Inventory:[line break]";
	list the contents of the player, with newlines, indented, giving inventory information, including all contents, with extra indentation.

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Welcome to TextWorld! You find yourself in a messy house. Many things are not in their usual location. Let's clean up this place. After you'll have done, this little house is going to be spick and spa".
The objective part 1 is some text that varies. The objective part 1 is "n! Look for anything that is out of place and put it away in its proper location.".

An objective is some text that varies. The objective is "[objective part 0][objective part 1]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

