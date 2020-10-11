from pathlib import Path
import sys
import networkx as nx
import logging
from textworld.logic import State, Rule, Proposition, Variable
from textworld import Game
from functools import lru_cache

# Some of these functions are from https://github.com/xingdi-eric-yuan/GATA-public

# Logging formatting
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level='INFO', stream=sys.stdout)

##############################
# KG stuff
##############################
# relations
two_args_relations = ["in", "on", "at", "west_of", "east_of", "north_of", "south_of", "part_of", "needs"]
one_arg_state_relations = ["chopped", "roasted", "diced", "burned", "open", "fried", "grilled", "consumed", "closed", "sliced", "uncut", "raw"]
ignore_relations = ["cuttable", "edible", "drinkable", "sharp", "inedible", "cut", "cooked", "cookable",
                    "needs_cooking"]
opposite_relations = {"west_of": "east_of",
                      "east_of": "west_of",
                      "south_of": "north_of",
                      "north_of": "south_of"}
equivalent_entities = {"inventory": "player",
                       "recipe": "cookbook"}
FOOD_FACTS = ["sliced", "diced", "chopped", "cut", "uncut", "cooked", "burned",
              "grilled", "fried", "roasted", "raw", "edible", "inedible"]

@lru_cache()
def _rules_predicates_scope():
    rules = [
        Rule.parse("query :: at(P, r) -> at(P, r)"),
        Rule.parse("query :: at(P, r) & at(o, r) -> at(o, r)"),
        Rule.parse("query :: at(P, r) & at(d, r) -> at(d, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) -> at(s, r)"),
        Rule.parse("query :: at(P, r) & at(c, r) -> at(c, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) & on(o, s) -> on(o, s)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) -> open(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & closed(c) -> closed(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(o, c) -> in(o, c)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & open(d) -> open(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & closed(d) -> closed(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    rules += [Rule.parse("query :: at(P, r) & at(f, r) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(s, r) & on(f, s) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(f, c) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_exits():
    rules = [
        Rule.parse("query :: at(P, r) & north_of(r', r) -> north_of(r', r)"),
        Rule.parse("query :: at(P, r) & west_of(r', r) -> west_of(r', r)"),
        Rule.parse("query :: at(P, r) & south_of(r', r) -> south_of(r', r)"),
        Rule.parse("query :: at(P, r) & east_of(r', r) -> east_of(r', r)"),
    ]
    return rules

@lru_cache()
def _rules_predicates_recipe():
    rules = [
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) -> part_of(f, RECIPE)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & roasted(ingredient) -> needs_roasted(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & grilled(ingredient) -> needs_grilled(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & fried(ingredient) -> needs_fried(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & sliced(ingredient) -> needs_sliced(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & chopped(ingredient) -> needs_chopped(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & diced(ingredient) -> needs_diced(f)"),
    ]
    return rules


@lru_cache()
def _rules_to_convert_link_predicates():
    rules = [
        Rule.parse("query :: link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    return rules


@lru_cache()
def _rules_predicates_inv():
    rules = [
        Rule.parse("query :: in(o, I) -> in(o, I)"),
    ]
    rules += [Rule.parse("query :: in(f, I) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


def convert_link_predicates(state):
    actions = state.all_applicable_actions(_rules_to_convert_link_predicates())
    for action in list(actions):
        state.apply(action)
    return state


def find_predicates_in_inventory(state):
    actions = state.all_applicable_actions(_rules_predicates_inv())
    return [action.postconditions[0] for action in actions]


def find_predicates_in_recipe(state):
    actions = state.all_applicable_actions(_rules_predicates_recipe())

    def _convert_to_needs_relation(proposition):
        if not proposition.name.startswith("needs_"):
            return proposition

        return Proposition("needs",
                           [proposition.arguments[0],
                            Variable(proposition.name.split("needs_")[-1], "STATE")])

    return [_convert_to_needs_relation(action.postconditions[0]) for action in actions]


def find_predicates_in_scope(state):
    actions = state.all_applicable_actions(_rules_predicates_scope())
    return [action.postconditions[0] for action in actions]


def find_exits_in_scope(state):
    actions = state.all_applicable_actions(_rules_exits())

    def _convert_to_exit_fact(proposition):
        return Proposition(proposition.name,
                           [Variable("exit", "LOCATION"),
                            proposition.arguments[1],
                            proposition.arguments[0]])

    return [_convert_to_exit_fact(action.postconditions[0]) for action in actions]


def process_direction_triplets(list_of_triplets):
    res = []
    for t in list_of_triplets:
        res.append(t)
        if t[0] == "exit" or t[1] == "exit":
            continue
        if "north_of" in t:
            res.append([t[1], t[0], "south_of"])
        elif "south_of" in t:
            res.append([t[1], t[0], "north_of"])
        elif "east_of" in t:
            res.append([t[1], t[0], "west_of"])
        elif "west_of" in t:
            res.append([t[1], t[0], "east_of"])
    return res


def process_equivalent_entities_in_triplet(triplet):
    # ["cookbook", "inventory", "in"]
    for i in range(len(triplet)):
        if triplet[i] in equivalent_entities:
            triplet[i] = equivalent_entities[triplet[i]]
    return triplet


def process_exits_in_triplet(triplet):
    # ["exit", "kitchen", "backyard", "south_of"]
    if triplet[0] == "exit":
        return [triplet[0], triplet[1], triplet[3]]
    else:
        return triplet


def process_burning_triplets(list_of_triplets):
    burned_stuff = []
    for t in list_of_triplets:
        if "burned" in t:
            burned_stuff.append(t[0])
    res = []
    for t in list_of_triplets:
        if t[0] in burned_stuff and t[1] in ["grilled", "fried", "roasted"]:
            continue
        res.append(t)
    return res


def serialize_facts(facts):
    PREDICATES_TO_DISCARD = {"ingredient_1", "ingredient_2", "ingredient_3", "ingredient_4", "ingredient_5",
                             "out", "free", "used", "cooking_location", "link"}
    CONSTANT_NAMES = {"P": "player", "I": "player", "ingredient": None, "slot": None, "RECIPE": "cookbook"}
    serialized_facts = [
        [arg.name if arg.name and arg.type not in CONSTANT_NAMES else CONSTANT_NAMES[arg.type] for arg in
         fact.arguments] + [fact.name]
        for fact in sorted(facts) if fact.name not in PREDICATES_TO_DISCARD]
    serialized_facts = [fact for fact in serialized_facts if None not in fact]

    current_facts = []
    unknown_relations = []
    for fact in serialized_facts:
        # fact = process_equivalent_entities_in_triplet(fact)
        fact = process_exits_in_triplet(fact)
        if fact[-1] in (two_args_relations + one_arg_state_relations):
            # known relations
            current_facts.append([it.lower() for it in fact])
        else:
            current_facts.append([it.lower() for it in fact])
            unknown_relations.append(fact[0])

    for fact in current_facts:
        if len(fact) == 2:
            fact.append("is")
        else:
            fact[1:-1] = [' '.join(fact[1:-1])]
            # print("Error: Found {} component fact instead of 3".format(len(fact)))
    current_facts = process_burning_triplets(current_facts)
    # current_facts = process_direction_triplets(current_facts)
    return current_facts


def process_step_facts(prev_facts, info_game, info_facts, info_last_action, cmd):
    kb = info_game.kb

    if prev_facts is None or cmd == "restart" or len(prev_facts)==0:
        new_facts = set()
    else:
        if cmd == "inventory": # Bypassing TextWorld's action detection.
            new_facts = set(find_predicates_in_inventory(State(kb.logic, info_facts)))
            return prev_facts | new_facts

        elif info_last_action is None:
            return prev_facts  # Invalid action, nothing has changed.

        # Check recipe from cookbook here
        elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
            new_facts = set(find_predicates_in_recipe(State(kb.logic, info_facts)))
            return prev_facts | new_facts

        state = State(kb.logic, prev_facts | set(info_last_action.preconditions))
        success = state.apply(info_last_action)
        assert success
        new_facts = set(state.facts)

    new_facts |= set(find_predicates_in_scope(State(kb.logic, info_facts)))
    new_facts |= set(find_exits_in_scope(State(kb.logic, info_facts)))
    return new_facts


def process_full_facts(info_game, facts):
    state = State(info_game.kb.logic, facts)
    state = convert_link_predicates(state) # applies all applicable actions for fully observable graph
    inventory_facts = set(find_predicates_in_inventory(state))
    # recipe_facts = set(find_predicates_in_recipe(state))
    return set(state.facts) | inventory_facts #| recipe_facts


def get_goal_graph(path):
    """
    :param path: path to any of the game files or file name without extension
    :returns: a nx.DiGraph that links every object in the game to its target locations
    """
    graph = nx.DiGraph()
    path = Path(path)
    if not path.exists():
        return None
    path = path.with_suffix('.json')
    game = Game.load(path)
    if 'goal_locations' not in game.metadata:
        return None
    goal_locations = game.metadata['goal_locations']
    for obj, locations in goal_locations.items():
        for loc in locations:
            graph.add_edge(obj, loc)
    return graph


def extract_entities(games):
    result = set()
    for g in games:
        for entity in g.infos.values():
            if entity.name:
                result.add((entity.name, entity.type))
    return result


def load_games(path):
    path = Path(path)
    files = path.rglob("*.json")
    return [Game.load(str(f)) for f in files]


