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

The internal name of r_0 is "laundry room".
The printed name of r_0 is "-= Laundry Room =-".
The laundry room part 0 is some text that varies. The laundry room part 0 is "You are in a laundry room. A typical kind of place.

 You rest your hand against a wall, but you miss the wall and fall onto a washing machine.[if c_1 is open and there is something in the c_1] The washing machine contains [a list of things in the c_1].[end if]".
The laundry room part 1 is some text that varies. The laundry room part 1 is "[if c_1 is open and the c_1 contains nothing] What a letdown! The washing machine is empty![end if]".
The laundry room part 2 is some text that varies. The laundry room part 2 is " You make out [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The laundry room part 3 is some text that varies. The laundry room part 3 is " clothes drier.[if c_2 is open and there is something in the c_2] The clothes drier contains [a list of things in the c_2], so there's that.[end if]".
The laundry room part 4 is some text that varies. The laundry room part 4 is "[if c_2 is open and the c_2 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The laundry room part 5 is some text that varies. The laundry room part 5 is " You see a laundry basket.[if c_5 is open and there is something in the c_5] The laundry basket contains [a list of things in the c_5]. The light flickers for a second, but nothing else happens.[end if]".
The laundry room part 6 is some text that varies. The laundry room part 6 is "[if c_5 is open and the c_5 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The laundry room part 7 is some text that varies. The laundry room part 7 is " You make out a work table. [if there is something on the s_10]On the work table you can make out [a list of things on the s_10].[end if]".
The laundry room part 8 is some text that varies. The laundry room part 8 is "[if there is nothing on the s_10]Unfortunately, there isn't a thing on it. Aw, and here you were, all excited for there to be things on it![end if]".
The laundry room part 9 is some text that varies. The laundry room part 9 is " You see a bench. The bench is w.[if there is something on the s_4] On the bench you can make out [a list of things on the s_4]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The laundry room part 10 is some text that varies. The laundry room part 10 is "[if there is nothing on the s_4] But there isn't a thing on it.[end if]".
The laundry room part 11 is some text that varies. The laundry room part 11 is " You rest your hand against a wall, but you miss the wall and fall onto a suspended shelf. [if there is something on the s_9]You see [a list of things on the s_9] on the suspended shelf. Wow! Just like in the movies![end if]".
The laundry room part 12 is some text that varies. The laundry room part 12 is "[if there is nothing on the s_9]Looks like someone's already been here and taken everything off it, though.[end if]".
The laundry room part 13 is some text that varies. The laundry room part 13 is "

There is an exit to the west. Don't worry, there is no door.".
The description of r_0 is "[laundry room part 0][laundry room part 1][laundry room part 2][laundry room part 3][laundry room part 4][laundry room part 5][laundry room part 6][laundry room part 7][laundry room part 8][laundry room part 9][laundry room part 10][laundry room part 11][laundry room part 12][laundry room part 13]".

The r_1 is mapped west of r_0.
The internal name of r_1 is "bathroom".
The printed name of r_1 is "-= Bathroom =-".
The bathroom part 0 is some text that varies. The bathroom part 0 is "Of every bathroom you could have shown up in, you had to show up in an ordinary one. You begin looking for stuff.

 You scan the room for a shower, and you find a shower.[if c_0 is open and there is something in the c_0] The shower contains [a list of things in the c_0].[end if]".
The bathroom part 1 is some text that varies. The bathroom part 1 is "[if c_0 is open and the c_0 contains nothing] The shower is empty, what a horrible day![end if]".
The bathroom part 2 is some text that varies. The bathroom part 2 is " You scan the room, seeing a bathroom cabinet.[if c_3 is open and there is something in the c_3] The bathroom cabinet contains [a list of things in the c_3].[end if]".
The bathroom part 3 is some text that varies. The bathroom part 3 is "[if c_3 is open and the c_3 contains nothing] The bathroom cabinet is empty, what a horrible day![end if]".
The bathroom part 4 is some text that varies. The bathroom part 4 is " You can see a pedal bin.[if c_4 is open and there is something in the c_4] The pedal bin contains [a list of things in the c_4].[end if]".
The bathroom part 5 is some text that varies. The bathroom part 5 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bathroom part 6 is some text that varies. The bathroom part 6 is " You scan the room, seeing a toilet. The toilet is white.[if there is something on the s_0] On the toilet you can see [a list of things on the s_0]. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The bathroom part 7 is some text that varies. The bathroom part 7 is "[if there is nothing on the s_0] The toilet appears to be empty. What, you think everything in TextWorld should have stuff on it?[end if]".
The bathroom part 8 is some text that varies. The bathroom part 8 is " You can make out a sink. [if there is something on the s_1]You see [a list of things on the s_1] on the sink.[end if]".
The bathroom part 9 is some text that varies. The bathroom part 9 is "[if there is nothing on the s_1]But the thing is empty.[end if]".
The bathroom part 10 is some text that varies. The bathroom part 10 is " You see a toilet roll holder. [if there is something on the s_2]On the toilet roll holder you can make out [a list of things on the s_2].[end if]".
The bathroom part 11 is some text that varies. The bathroom part 11 is "[if there is nothing on the s_2]However, the toilet roll holder, like an empty toilet roll holder, has nothing on it. Aw, here you were, all excited for there to be things on it![end if]".
The bathroom part 12 is some text that varies. The bathroom part 12 is " Hey, want to see a towel rail? Look over there, a towel rail. [if there is something on the s_3]You see [a list of things on the s_3] on the towel rail.[end if]".
The bathroom part 13 is some text that varies. The bathroom part 13 is "[if there is nothing on the s_3]But the thing is empty, unfortunately. What, you think everything in TextWorld should have stuff on it?[end if]".
The bathroom part 14 is some text that varies. The bathroom part 14 is " Oh, great. Here's a bathtub! The bathtub is typical.[if there is something on the s_5] On the bathtub you can make out [a list of things on the s_5]. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The bathroom part 15 is some text that varies. The bathroom part 15 is "[if there is nothing on the s_5] But there isn't a thing on it.[end if]".
The bathroom part 16 is some text that varies. The bathroom part 16 is " Hey, want to see a bath mat? Look over there, a bath mat. [if there is something on the s_6]You see [a list of things on the s_6] on the bath mat.[end if]".
The bathroom part 17 is some text that varies. The bathroom part 17 is "[if there is nothing on the s_6]Looks like someone's already been here and taken everything off it, though. What, you think everything in TextWorld should have stuff on it?[end if]".
The bathroom part 18 is some text that varies. The bathroom part 18 is " As if things weren't amazing enough already, you can even see a dressing table. The dressing table is typical.[if there is something on the s_7] On the dressing table you can see [a list of things on the s_7].[end if]".
The bathroom part 19 is some text that varies. The bathroom part 19 is "[if there is nothing on the s_7] Unfortunately, there isn't a thing on it.[end if]".
The bathroom part 20 is some text that varies. The bathroom part 20 is " You scan the room for a wall hook, and you find a wall hook. Wow, isn't TextWorld just the best? The wall hook is typical.[if there is something on the s_8] On the wall hook you make out [a list of things on the s_8].[end if]".
The bathroom part 21 is some text that varies. The bathroom part 21 is "[if there is nothing on the s_8] But the thing hasn't got anything on it.[end if]".
The bathroom part 22 is some text that varies. The bathroom part 22 is "

There is an exit to the east.".
The description of r_1 is "[bathroom part 0][bathroom part 1][bathroom part 2][bathroom part 3][bathroom part 4][bathroom part 5][bathroom part 6][bathroom part 7][bathroom part 8][bathroom part 9][bathroom part 10][bathroom part 11][bathroom part 12][bathroom part 13][bathroom part 14][bathroom part 15][bathroom part 16][bathroom part 17][bathroom part 18][bathroom part 19][bathroom part 20][bathroom part 21][bathroom part 22]".

The r_0 is mapped east of r_1.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The o_3 and the o_2 and the o_4 and the o_0 and the o_1 and the o_5 are object-likes.
The o_3 and the o_2 and the o_4 and the o_0 and the o_1 and the o_5 are privately-named.
The r_0 and the r_1 are rooms.
The r_0 and the r_1 are privately-named.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.
The slot_0 and the slot_1 and the slot_10 and the slot_11 and the slot_12 and the slot_13 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 are things.
The slot_0 and the slot_1 and the slot_10 and the slot_11 and the slot_12 and the slot_13 and the slot_2 and the slot_3 and the slot_4 and the slot_5 and the slot_6 and the slot_7 and the slot_8 and the slot_9 are privately-named.

The description of c_0 is "The [noun] looks well-built. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to force open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_0 is "shower".
Understand "shower" as c_0.
The c_0 is in r_1.
The c_0 is open.
The description of c_1 is "The [noun] looks noble. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to crack open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_1 is "washing machine".
Understand "washing machine" as c_1.
Understand "washing" as c_1.
Understand "machine" as c_1.
The c_1 is in r_0.
The c_1 is open.
The description of c_2 is "The [noun] looks sturdy. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to break open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_2 is "clothes drier".
Understand "clothes drier" as c_2.
Understand "clothes" as c_2.
Understand "drier" as c_2.
The c_2 is in r_0.
The c_2 is open.
The description of c_3 is "The [noun] looks durable. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to force open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_3 is "bathroom cabinet".
Understand "bathroom cabinet" as c_3.
Understand "bathroom" as c_3.
Understand "cabinet" as c_3.
The c_3 is in r_1.
The c_3 is open.
The description of c_4 is "The [noun] looks ominous. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_4 is "pedal bin".
Understand "pedal bin" as c_4.
Understand "pedal" as c_4.
Understand "bin" as c_4.
The c_4 is in r_1.
The c_4 is open.
The description of c_5 is "The [noun] looks manageable. [if open]You can see inside it.[else if locked]There is a lock on it and seems impossible to force open.[otherwise]You can't see inside it because the lid's in your way.[end if]".
The printed name of c_5 is "laundry basket".
Understand "laundry basket" as c_5.
Understand "laundry" as c_5.
Understand "basket" as c_5.
The c_5 is in r_0.
The c_5 is open.
The description of o_3 is "The [noun] would seem to be out of place here".
The printed name of o_3 is "wet yellow dress".
Understand "wet yellow dress" as o_3.
Understand "wet" as o_3.
Understand "yellow" as o_3.
Understand "dress" as o_3.
The o_3 is in r_1.
The description of s_0 is "The [noun] is reliable.".
The printed name of s_0 is "toilet".
Understand "toilet" as s_0.
The s_0 is in r_1.
The description of s_1 is "The [noun] is stable.".
The printed name of s_1 is "sink".
Understand "sink" as s_1.
The s_1 is in r_1.
The description of s_10 is "The [noun] is shaky.".
The printed name of s_10 is "work table".
Understand "work table" as s_10.
Understand "work" as s_10.
Understand "table" as s_10.
The s_10 is in r_0.
The description of s_2 is "The [noun] is durable.".
The printed name of s_2 is "toilet roll holder".
Understand "toilet roll holder" as s_2.
Understand "toilet" as s_2.
Understand "roll" as s_2.
Understand "holder" as s_2.
The s_2 is in r_1.
The description of s_3 is "The [noun] is solid.".
The printed name of s_3 is "towel rail".
Understand "towel rail" as s_3.
Understand "towel" as s_3.
Understand "rail" as s_3.
The s_3 is in r_1.
The description of s_4 is "The [noun] is stable.".
The printed name of s_4 is "bench".
Understand "bench" as s_4.
The s_4 is in r_0.
The description of s_5 is "The [noun] is wobbly.".
The printed name of s_5 is "bathtub".
Understand "bathtub" as s_5.
The s_5 is in r_1.
The description of s_6 is "The [noun] is unstable.".
The printed name of s_6 is "bath mat".
Understand "bath mat" as s_6.
Understand "bath" as s_6.
Understand "mat" as s_6.
The s_6 is in r_1.
The description of s_7 is "The [noun] is undependable.".
The printed name of s_7 is "dressing table".
Understand "dressing table" as s_7.
Understand "dressing" as s_7.
Understand "table" as s_7.
The s_7 is in r_1.
The description of s_8 is "The [noun] is balanced.".
The printed name of s_8 is "wall hook".
Understand "wall hook" as s_8.
Understand "wall" as s_8.
Understand "hook" as s_8.
The s_8 is in r_1.
The description of s_9 is "The [noun] is durable.".
The printed name of s_9 is "suspended shelf".
Understand "suspended shelf" as s_9.
Understand "suspended" as s_9.
Understand "shelf" as s_9.
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
The description of o_2 is "The [noun] appears well matched to everything else here".
The printed name of o_2 is "dirty black bra".
Understand "dirty black bra" as o_2.
Understand "dirty" as o_2.
Understand "black" as o_2.
Understand "bra" as o_2.
The o_2 is in the c_3.
The description of o_4 is "The [noun] is antiquated.".
The printed name of o_4 is "dirty yellow T-shirt".
Understand "dirty yellow T-shirt" as o_4.
Understand "dirty" as o_4.
Understand "yellow" as o_4.
Understand "T-shirt" as o_4.
The o_4 is in the c_2.
The description of f_0 is "That's a [noun]!".
The printed name of f_0 is "rotten yellow apple".
Understand "rotten yellow apple" as f_0.
Understand "rotten" as f_0.
Understand "yellow" as f_0.
Understand "apple" as f_0.
The f_0 is on the s_7.
The description of o_0 is "The [noun] seems to fit in here".
The printed name of o_0 is "wet cardigan".
Understand "wet cardigan" as o_0.
Understand "wet" as o_0.
Understand "cardigan" as o_0.
The o_0 is on the s_7.
The description of o_1 is "The [noun] is modern.".
The printed name of o_1 is "dirty checkered blazer".
Understand "dirty checkered blazer" as o_1.
Understand "dirty" as o_1.
Understand "checkered" as o_1.
Understand "blazer" as o_1.
The o_1 is on the s_4.
The description of o_5 is "The [noun] is clean.".
The printed name of o_5 is "dirty brown dress".
Understand "dirty brown dress" as o_5.
Understand "dirty" as o_5.
Understand "brown" as o_5.
Understand "dress" as o_5.
The o_5 is on the s_6.


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with ""

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The f_0 is in the c_4:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with ""

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The o_0 is in the c_2:
		increase the score by 1; [Quest completed]
		Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with ""


Test quest2_1 with ""

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The o_1 is in the c_1:
		increase the score by 1; [Quest completed]
		Now the quest2 completed is true;
	else if The o_1 is in the c_5:
		increase the score by 1; [Quest completed]
		Now the quest2 completed is true;

The quest3 completed is a truth state that varies.
The quest3 completed is usually false.

Test quest3_0 with ""


Test quest3_1 with ""

Every turn:
	if quest3 completed is true:
		do nothing;
	else if The o_2 is in the c_1:
		increase the score by 1; [Quest completed]
		Now the quest3 completed is true;
	else if The o_2 is in the c_5:
		increase the score by 1; [Quest completed]
		Now the quest3 completed is true;

The quest4 completed is a truth state that varies.
The quest4 completed is usually false.

Test quest4_0 with ""

Every turn:
	if quest4 completed is true:
		do nothing;
	else if The o_3 is in the c_2:
		increase the score by 1; [Quest completed]
		Now the quest4 completed is true;

The quest5 completed is a truth state that varies.
The quest5 completed is usually false.

Test quest5_0 with ""


Test quest5_1 with ""

Every turn:
	if quest5 completed is true:
		do nothing;
	else if The o_4 is in the c_1:
		increase the score by 1; [Quest completed]
		Now the quest5 completed is true;
	else if The o_4 is in the c_5:
		increase the score by 1; [Quest completed]
		Now the quest5 completed is true;

The quest6 completed is a truth state that varies.
The quest6 completed is usually false.

Test quest6_0 with ""


Test quest6_1 with ""

Every turn:
	if quest6 completed is true:
		do nothing;
	else if The o_5 is in the c_1:
		increase the score by 1; [Quest completed]
		Now the quest6 completed is true;
	else if The o_5 is in the c_5:
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

