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

The internal name of r_0 is "livingroom".
The printed name of r_0 is "-= Livingroom =-".
The livingroom part 0 is some text that varies. The livingroom part 0 is "You arrive in a livingroom. A typical one.

 You see a bookcase.[if c_1 is open and there is something in the c_1] The bookcase contains [a list of things in the c_1].[end if]".
The livingroom part 1 is some text that varies. The livingroom part 1 is "[if c_1 is open and the c_1 contains nothing] The bookcase is empty! What a waste of a day![end if]".
The livingroom part 2 is some text that varies. The livingroom part 2 is " You make out [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The livingroom part 3 is some text that varies. The livingroom part 3 is " wastepaper basket close by.[if c_3 is open and there is something in the c_3] The wastepaper basket contains [a list of things in the c_3].[end if]".
The livingroom part 4 is some text that varies. The livingroom part 4 is "[if c_3 is open and the c_3 contains nothing] The wastepaper basket is empty, what a horrible day![end if]".
The livingroom part 5 is some text that varies. The livingroom part 5 is " Look over there! a sofa. The sofa is comfy.[if there is something on the s_1] On the sofa you make out [a list of things on the s_1]. Huh, weird.[end if]".
The livingroom part 6 is some text that varies. The livingroom part 6 is "[if there is nothing on the s_1] But oh no! there's nothing on this piece of junk. What's the point of an empty sofa?[end if]".
The livingroom part 7 is some text that varies. The livingroom part 7 is " Were you looking for an armchair? Because look over there, it's an armchair. The armchair is comfortable.[if there is something on the s_11] On the armchair you can make out [a list of things on the s_11].[end if]".
The livingroom part 8 is some text that varies. The livingroom part 8 is "[if there is nothing on the s_11] The armchair appears to be empty.[end if]".
The livingroom part 9 is some text that varies. The livingroom part 9 is " You can make out a TV stand. The TV stand is ordinary.[if there is something on the s_12] On the TV stand you make out [a list of things on the s_12].[end if]".
The livingroom part 10 is some text that varies. The livingroom part 10 is "[if there is nothing on the s_12] The TV stand appears to be empty.[end if]".
The livingroom part 11 is some text that varies. The livingroom part 11 is " You smell an interesting smell, and follow it to a grey carpet. The grey carpet is standard.[if there is something on the s_4] On the grey carpet you can see [a list of things on the s_4].[end if]".
The livingroom part 12 is some text that varies. The livingroom part 12 is "[if there is nothing on the s_4] The grey carpet appears to be empty. Hm. Oh well[end if]".
The livingroom part 13 is some text that varies. The livingroom part 13 is " You make out a side table. [if there is something on the s_6]On the side table you can make out [a list of things on the s_6].[end if]".
The livingroom part 14 is some text that varies. The livingroom part 14 is "[if there is nothing on the s_6]But there isn't a thing on it. What, you think everything in TextWorld should have stuff on it?[end if]".
The livingroom part 15 is some text that varies. The livingroom part 15 is " You see a coffee table. [if there is something on the s_8]On the coffee table you can see [a list of things on the s_8].[end if]".
The livingroom part 16 is some text that varies. The livingroom part 16 is "[if there is nothing on the s_8]Looks like someone's already been here and taken everything off it, though.[end if]".
The livingroom part 17 is some text that varies. The livingroom part 17 is " You can make out an end table. The end table is wooden.[if there is something on the s_9] On the end table you can make out [a list of things on the s_9].[end if]".
The livingroom part 18 is some text that varies. The livingroom part 18 is "[if there is nothing on the s_9] Unfortunately, there isn't a thing on it.[end if]".
The livingroom part 19 is some text that varies. The livingroom part 19 is "

You need an exit without a door? You should try going west.".
The description of r_0 is "[livingroom part 0][livingroom part 1][livingroom part 2][livingroom part 3][livingroom part 4][livingroom part 5][livingroom part 6][livingroom part 7][livingroom part 8][livingroom part 9][livingroom part 10][livingroom part 11][livingroom part 12][livingroom part 13][livingroom part 14][livingroom part 15][livingroom part 16][livingroom part 17][livingroom part 18][livingroom part 19]".

The r_1 is mapped west of r_0.
The internal name of r_1 is "bedroom".
The printed name of r_1 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "You've entered a bedroom.

 You see a wardrobe.[if c_0 is open and there is something in the c_0] The wardrobe contains [a list of things in the c_0].[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is " You see a chest of drawers. Now why would someone leave that there?[if c_2 is open and there is something in the c_2] The chest of drawers contains [a list of things in the c_2]. Now why would someone leave that there?[end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is "[if c_2 is open and the c_2 contains nothing] The chest of drawers is empty! What a waste of a day![end if]".
The bedroom part 4 is some text that varies. The bedroom part 4 is " You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a bed. [if there is something on the s_0]You see [a list of things on the s_0] on the bed.[end if]".
The bedroom part 5 is some text that varies. The bedroom part 5 is "[if there is nothing on the s_0]Unfortunately, there isn't a thing on it.[end if]".
The bedroom part 6 is some text that varies. The bedroom part 6 is " You bend down to tie your shoe. When you stand up, you notice a nightstand. The nightstand is ordinary.[if there is something on the s_10] On the nightstand you see [a list of things on the s_10].[end if]".
The bedroom part 7 is some text that varies. The bedroom part 7 is "[if there is nothing on the s_10] Unfortunately, there isn't a thing on it. Hm. Oh well[end if]".
The bedroom part 8 is some text that varies. The bedroom part 8 is " Look out! It's a- oh, never mind, it's just a dressing table. [if there is something on the s_2]On the dressing table you can see [a list of things on the s_2].[end if]".
The bedroom part 9 is some text that varies. The bedroom part 9 is "[if there is nothing on the s_2]Unfortunately, there isn't a thing on it. What, you think everything in TextWorld should have stuff on it?[end if]".
The bedroom part 10 is some text that varies. The bedroom part 10 is " You can make out a desk chair. The desk chair is usual.[if there is something on the s_3] On the desk chair you can make out [a list of things on the s_3]. Now that's what I call TextWorld![end if]".
The bedroom part 11 is some text that varies. The bedroom part 11 is "[if there is nothing on the s_3] But the thing is empty. Silly desk chair, silly, empty, good for nothing desk chair.[end if]".
The bedroom part 12 is some text that varies. The bedroom part 12 is " You see a desk. [if there is something on the s_5]On the desk you can see [a list of things on the s_5].[end if]".
The bedroom part 13 is some text that varies. The bedroom part 13 is "[if there is nothing on the s_5]But oh no! there's nothing on this piece of garbage. Hm. Oh well[end if]".
The bedroom part 14 is some text that varies. The bedroom part 14 is " Hey, want to see a dark carpet? Look over there, a dark carpet. [if there is something on the s_7]On the dark carpet you see [a list of things on the s_7].[end if]".
The bedroom part 15 is some text that varies. The bedroom part 15 is "[if there is nothing on the s_7]The dark carpet appears to be empty.[end if]".
The bedroom part 16 is some text that varies. The bedroom part 16 is "

You need an exit without a door? You should try going east.".
The description of r_1 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3][bedroom part 4][bedroom part 5][bedroom part 6][bedroom part 7][bedroom part 8][bedroom part 9][bedroom part 10][bedroom part 11][bedroom part 12][bedroom part 13][bedroom part 14][bedroom part 15][bedroom part 16]".

The r_0 is mapped east of r_1.

The c_0 and the c_1 and the c_2 and the c_3 are containers.
The c_0 and the c_1 and the c_2 and the c_3 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The o_3 and the o_5 and the o_0 and the o_4 and the o_1 and the o_2 are object-likes.
The o_3 and the o_5 and the o_0 and the o_4 and the o_1 and the o_2 are privately-named.
The r_0 and the r_1 are rooms.
The r_0 and the r_1 are privately-named.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.
The slot_0 and the slot_1 and the slot_10 and the slot_11 and the slot_12 and the slot_13 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 are things.
The slot_0 and the slot_1 and the slot_10 and the slot_11 and the slot_12 and the slot_13 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 are privately-named.

The description of c_0 is "The [noun] looks robust. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to bash open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_0 is "wardrobe".
Understand "wardrobe" as c_0.
The c_0 is in r_1.
The c_0 is open.
The description of c_1 is "The [noun] looks stuffy. [if open]It is open.[else if locked]It is locked.[otherwise]It is closed.[end if]".
The printed name of c_1 is "bookcase".
Understand "bookcase" as c_1.
The c_1 is in r_0.
The c_1 is open.
The description of c_2 is "The [noun] looks hefty. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to break open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_2 is "chest of drawers".
Understand "chest of drawers" as c_2.
Understand "chest" as c_2.
Understand "drawers" as c_2.
The c_2 is in r_1.
The c_2 is open.
The description of c_3 is "The [noun] looks ominous. [if open]It is open.[else if locked]It is locked.[otherwise]It is closed.[end if]".
The printed name of c_3 is "wastepaper basket".
Understand "wastepaper basket" as c_3.
Understand "wastepaper" as c_3.
Understand "basket" as c_3.
The c_3 is in r_0.
The c_3 is open.
The description of o_3 is "The [noun] seems out of place here".
The printed name of o_3 is "fleece jacket".
Understand "fleece jacket" as o_3.
Understand "fleece" as o_3.
Understand "jacket" as o_3.
The o_3 is in r_1.
The description of o_5 is "The [noun] is dirty.".
The printed name of o_5 is "blue tie".
Understand "blue tie" as o_5.
Understand "blue" as o_5.
Understand "tie" as o_5.
The o_5 is in r_1.
The description of s_0 is "The [noun] is solid.".
The printed name of s_0 is "bed".
Understand "bed" as s_0.
The s_0 is in r_1.
The description of s_1 is "The [noun] is stable.".
The printed name of s_1 is "sofa".
Understand "sofa" as s_1.
The s_1 is in r_0.
The description of s_10 is "The [noun] is reliable.".
The printed name of s_10 is "nightstand".
Understand "nightstand" as s_10.
The s_10 is in r_1.
The description of s_11 is "The [noun] is unstable.".
The printed name of s_11 is "armchair".
Understand "armchair" as s_11.
The s_11 is in r_0.
The description of s_12 is "The [noun] is solid.".
The printed name of s_12 is "TV stand".
Understand "TV stand" as s_12.
Understand "TV" as s_12.
Understand "stand" as s_12.
The s_12 is in r_0.
The description of s_2 is "The [noun] is wobbly.".
The printed name of s_2 is "dressing table".
Understand "dressing table" as s_2.
Understand "dressing" as s_2.
Understand "table" as s_2.
The s_2 is in r_1.
The description of s_3 is "The [noun] is unstable.".
The printed name of s_3 is "desk chair".
Understand "desk chair" as s_3.
Understand "desk" as s_3.
Understand "chair" as s_3.
The s_3 is in r_1.
The description of s_4 is "The [noun] is solid.".
The printed name of s_4 is "grey carpet".
Understand "grey carpet" as s_4.
Understand "grey" as s_4.
Understand "carpet" as s_4.
The s_4 is in r_0.
The description of s_5 is "The [noun] is unstable.".
The printed name of s_5 is "desk".
Understand "desk" as s_5.
The s_5 is in r_1.
The description of s_6 is "The [noun] is balanced.".
The printed name of s_6 is "side table".
Understand "side table" as s_6.
Understand "side" as s_6.
Understand "table" as s_6.
The s_6 is in r_0.
The description of s_7 is "The [noun] is durable.".
The printed name of s_7 is "dark carpet".
Understand "dark carpet" as s_7.
Understand "dark" as s_7.
Understand "carpet" as s_7.
The s_7 is in r_1.
The description of s_8 is "The [noun] is stable.".
The printed name of s_8 is "coffee table".
Understand "coffee table" as s_8.
Understand "coffee" as s_8.
Understand "table" as s_8.
The s_8 is in r_0.
The description of s_9 is "The [noun] is solid.".
The printed name of s_9 is "end table".
Understand "end table" as s_9.
Understand "end" as s_9.
Understand "table" as s_9.
The s_9 is in r_0.
The description of slot_0 is "".
The printed name of slot_0 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_1 is "".
The printed name of slot_1 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_10 is "".
The printed name of slot_10 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_11 is "".
The printed name of slot_11 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_12 is "".
The printed name of slot_12 is "".
When play begins, increase the carrying capacity of the player by 1..
The description of slot_13 is "".
The printed name of slot_13 is "".
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
The description of o_0 is "The [noun] appears out of place here".
The printed name of o_0 is "clean white blazer".
Understand "clean white blazer" as o_0.
Understand "clean" as o_0.
Understand "white" as o_0.
Understand "blazer" as o_0.
The o_0 is in the c_2.
The description of o_4 is "The [noun] looks to fit in here".
The printed name of o_4 is "clean brown shirt".
Understand "clean brown shirt" as o_4.
Understand "clean" as o_4.
Understand "brown" as o_4.
Understand "shirt" as o_4.
The o_4 is in the c_1.
The description of f_0 is "The [noun] looks appealing.".
The printed name of f_0 is "rotten red apple".
Understand "rotten red apple" as f_0.
Understand "rotten" as f_0.
Understand "red" as f_0.
Understand "apple" as f_0.
The f_0 is on the s_11.
The description of o_1 is "The [noun] is clean.".
The printed name of o_1 is "teapot".
Understand "teapot" as o_1.
The o_1 is on the s_9.
The description of o_2 is "The [noun] is expensive looking.".
The printed name of o_2 is "clean gray blazer".
Understand "clean gray blazer" as o_2.
Understand "clean" as o_2.
Understand "gray" as o_2.
Understand "blazer" as o_2.
The o_2 is on the s_1.


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with ""

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The o_0 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with ""

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The f_0 is in the c_3:
		increase the score by 1; [Quest completed]
		Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with ""

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The o_1 is on the s_8:
		increase the score by 1; [Quest completed]
		Now the quest2 completed is true;

The quest3 completed is a truth state that varies.
The quest3 completed is usually false.

Test quest3_0 with ""

Every turn:
	if quest3 completed is true:
		do nothing;
	else if The o_2 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest3 completed is true;

The quest4 completed is a truth state that varies.
The quest4 completed is usually false.

Test quest4_0 with ""

Every turn:
	if quest4 completed is true:
		do nothing;
	else if The o_3 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest4 completed is true;

The quest5 completed is a truth state that varies.
The quest5 completed is usually false.

Test quest5_0 with ""

Every turn:
	if quest5 completed is true:
		do nothing;
	else if The o_4 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest5 completed is true;

The quest6 completed is a truth state that varies.
The quest6 completed is usually false.

Test quest6_0 with ""


Test quest6_1 with ""

Every turn:
	if quest6 completed is true:
		do nothing;
	else if The o_5 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest6 completed is true;
	else if The o_5 is in the c_2:
		increase the score by 1; [Quest completed]
		Now the quest6 completed is true;

Use scoring. The maximum score is 7.
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

