from iqa_cleanup_config import config
from iqa_cleanup_entities import FURNITURE, OBJECTS, ENTITIES
import random
from iqa_cleanup import illegal_locations
import gensim.downloader
from word_embedding_utils import precompute_embeddings
from pathlib import Path
from gensim.models import KeyedVectors
from functools import partial
import pandas as pd

POSITIVE_LABEL = '+'
NEGATIVE_LABEL = '-'
MIN_ANNOTATORS = 3
NUM_ANNOTATORS = 10
INSTANCES_PER_ANNOTATOR = 100
POSITIVE_PROB = 0.7
COLUMNS = ['entity', 'room', 'location', 'label']
OUTPUT_DIR = Path('annotation')
EMBEDDING_PATH = Path('iqa_dataset/entity_embeddings.txt')
LABELED_OUTPUT_FILE_FORMAT = 'annotator%03d_reference.csv'
OUTPUT_FILE_FORMAT = 'annotator%03d.csv'


def random_sample_positives(size):
    all_objects = list(OBJECTS.keys())
    random.shuffle(all_objects)
    object_list = all_objects[:size]
    result = []
    for o in object_list:
        locations = OBJECTS[o]["locations"]
        loc = random.choice(locations)
        if "." in loc:
            room = loc.split(".")[0]
            holder = loc.split(".")[1]
            result.append((o, room, holder))
        else:
            holder = loc
            room = random.choice(FURNITURE[holder]["locations"])
            result.append((o, room, holder))
    return result


def random_sample_negatives(size):
    all_objects = list(OBJECTS.keys())
    random.shuffle(all_objects)
    object_list = all_objects[:size]
    result = []

    for o in object_list:
        locations = {loc.split(".")[-1] for loc in OBJECTS[o]["locations"]}
        pool = set(FURNITURE.keys()) - locations - illegal_locations(o)
        holder = random.choice(pool)

        rooms = FURNITURE[holder]["locations"]
        room = random.choice(rooms)
        result.append((o, room, holder))

    return result


def sample_negatives_with_embeddings(size, emb):
    all_objects = list(OBJECTS.keys())
    random.shuffle(all_objects)
    object_list = all_objects[:size]
    result = []
    step = 30

    for o in object_list:
        locations = [loc.split(".")[-1] for loc in OBJECTS[o]["locations"]]
        correct_holder = random.choice(locations)
        correct_holder = correct_holder.strip().replace(' ', '_')
        pool = set()
        k = step
        while len(pool) < 10:
            most_similar = emb.most_similar(correct_holder, topn=k)
            most_similar = list(zip(*most_similar))[0]
            most_similar = set(most_similar) - set(locations) - illegal_locations(o)
            pool = [e for e in most_similar if e in FURNITURE]
            k += step

        pool = sorted(list(pool))
        holder = random.choice(pool)
        holder = holder.replace('_', ' ')
        rooms = FURNITURE[holder]["locations"]
        room = random.choice(rooms)
        result.append((o, room, holder))

    return result


def random_pick_positive():
    return random_sample_positives(1)[0]


def random_pick_negative():
    return random_sample_negatives(1)[0]


def pick_negative_with_embeddings(emb):
    return sample_negatives_with_embeddings(1, emb)[0]


def load_embeddings():
    if EMBEDDING_PATH.exists():
        return KeyedVectors.load_word2vec_format(EMBEDDING_PATH)

    print("Loading GloVe")
    glove = gensim.downloader.load("glove-wiki-gigaword-300")
    print("Done")

    return precompute_embeddings(glove, ENTITIES.keys(), EMBEDDING_PATH)


def generate_annotation_tuples(nb_annotators, size, positive_prob):
    positive_samples = round(positive_prob * size)
    negative_samples = size - positive_samples

    emb = load_embeddings()
    negative_picker = partial(pick_negative_with_embeddings, emb=emb)

    positive_assignments = pick_assignments(nb_annotators, positive_samples, random_pick_positive)
    negative_assignments = pick_assignments(nb_annotators, negative_samples, negative_picker)

    positive_assignments = label_assignments(positive_assignments, POSITIVE_LABEL)
    negative_assignments = label_assignments(negative_assignments, NEGATIVE_LABEL)

    merged = merge_assignments(positive_assignments, negative_assignments)

    return shuffle_assignments(merged)


def merge_assignments(a_1, a_2):
    merged = {k: set(v) for k, v in a_1.items()}
    for k, v in a_2.items():
        if k in merged:
            merged[k] |= set(v)
        else:
            merged[k] = set(v)
    return merged


def shuffle_assignments(assignments):
    result = {}
    for annotator in assignments:
        tuples = list(assignments[annotator])
        random.shuffle(tuples)
        result[annotator] = tuples
    return result


def label_assignments(assignments, label):
    result = {}
    for annotator in assignments:
        assigned_tuples = assignments[annotator]
        labeled_tuples = {t + (label, ) for t in assigned_tuples}
        result[annotator] = labeled_tuples
    return result


def pick_assignments(nb_annotators, size, picker):
    annotators = list(range(1, nb_annotators + 1))
    assignments = {i: set() for i in annotators}
    available_annotators = list(annotators)

    while len(available_annotators) >= MIN_ANNOTATORS:
        random.shuffle(available_annotators)
        assigned_annotators = available_annotators[:MIN_ANNOTATORS]
        picked_tuple = picker()
        for i in assigned_annotators:
            assignments[i].add(picked_tuple)
        available_annotators = [i for i in annotators if len(assignments[i]) < size]

    fill_assignments(assignments, size)

    return assignments


def fill_assignments(assignments, size):
    all_tuples = set.union(*assignments.values())
    for annotator in assignments:
        diff = size - len(assignments[annotator])
        assert diff >= 0
        if diff == 0:
            continue
        pool = list(all_tuples)
        selected = set()
        while len(selected) != diff:
            choice = random.choice(pool)
            if choice not in assignments[annotator]:
                selected.add(choice)
        assignments[annotator] |= selected


def _test_same_length(assignments):
    lengths = {len(l) for l in assignments.values()}
    return len(lengths) == 1


def _test_min_annotators(assignments):
    all_tuples = {t for tuples in assignments.values() for t in tuples}
    for t in all_tuples:
        count = 0
        for annotator in assignments:
            if t in assignments[annotator]:
                count += 1
        if count < MIN_ANNOTATORS:
            return False
    return True


def _sanity_check(assignments):
    assert _test_same_length(assignments), \
        "Annotators do not have the same number of assigned instances"
    assert _test_min_annotators(assignments),\
        f"Some instances are assigned to less than {MIN_ANNOTATORS} annotators"


def save_assignments(assignments):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for annotator in assignments:
        df = pd.DataFrame(assignments[annotator], columns=COLUMNS)
        labeled_file_name = LABELED_OUTPUT_FILE_FORMAT % annotator
        unlabeled_file_name = OUTPUT_FILE_FORMAT % annotator
        labeled_output_path = Path(OUTPUT_DIR) / labeled_file_name
        unlabeled_output_path = Path(OUTPUT_DIR) / unlabeled_file_name
        df.to_csv(labeled_output_path, index=False)
        df[COLUMNS[:-1]].to_csv(unlabeled_output_path, index=False)


def main():
    random.seed(config.seed)
    assignments = generate_annotation_tuples(NUM_ANNOTATORS, INSTANCES_PER_ANNOTATOR, POSITIVE_PROB)
    _sanity_check(assignments)
    save_assignments(assignments)
    print('Assignments saved successfully')


if __name__ == "__main__":
    main()
