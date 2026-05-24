import json
import os
from tqdm import tqdm
from copy import deepcopy
import random as rnd

rnd.seed(42)

def sro(rel, passage):
    passage = passage.lower()
    rel = rel.lower()
    return rel in passage


def so(models_results):
    return list(models_results.values()).count('yes') == 0


def support(models_results):
    return list(models_results.values()).count('yes') == 3


def categorize(model):
    categories = dict()
    questions_keys = deepcopy(list(dataset[model].keys()))
    rnd.shuffle(questions_keys)
    for qid in questions_keys:
        rel = dataset[model][qid]['rel']
        if rel not in categories:
            categories[rel] = dict()
        categories[rel][qid] = dataset[model][qid]
    return categories


def validate_questions(questions, model, rel, reserved):
    for qid in questions:
        stored_passages = {'sro': {}, 'so': {}, 'support': {}}
        if qid in reserved:
            continue
        with open(f'../../dataset/passages/{model}/{qid}.json') as f:
            passages = json.load(f)
        for passage in passages:
            if sum([len(val) for val in stored_passages.values()]) >= 15:
                return qid, stored_passages
            if sro(rel, passage):
                if len(stored_passages['sro']) < 5:
                    stored_passages['sro'][passage] = passages[passage]['relevance']
            if so(passages[passage]['models']):
                if len(stored_passages['so']) < 5:
                    stored_passages['so'][passage] = passages[passage]['relevance']
            if support(passages[passage]['models']):
                if len(stored_passages['support']) < 5:
                    stored_passages['support'][passage] = passages[passage]['relevance']
    return None, None


def ensure_required_data() -> bool:
    missing = []
    dataset_path = '../../dataset/exposureQA.json'
    passages_dir = '../../dataset/passages'

    if not os.path.isfile(dataset_path):
        missing.append(dataset_path)
    if not os.path.isdir(passages_dir):
        missing.append(passages_dir)

    if missing:
        print('ERROR: Missing required ExposureQA files/directories:')
        for path in missing:
            print(f'- {path}')
        print()
        print('What to do:')
        print('1) Go to the dataset directory of this project.')
        print('2) Run `download.sh`.')
        print('3) Select `ExposureQA-Full` mode to download the dataset and passages.')
        return False

    return True


if __name__ == "__main__":
    if ensure_required_data():
        with open('../../dataset/exposureQA.json', 'r') as f:
            dataset = json.load(f)
        samples = dict()
        reserved = []
        for i, model in enumerate(dataset):
            categories = categorize(model)
            qids = dict()
            for rel in tqdm(categories, desc=model):
                questions = categories[rel]
                qid, passages = validate_questions(questions, model, rel, reserved)
                qids[qid] = passages
                qids[qid]['sub'] = dataset[model][qid]['sub']
                qids[qid]['rel'] = dataset[model][qid]['rel']
                qids[qid]['obj'] = dataset[model][qid]['obj']
                reserved.append(qid)
            for j in tqdm(range(i * 5, (i + 1) * 5), desc=model):
                categories_keys = list(categories.keys())
                rel = categories_keys[j]
                questions = categories[rel]
                qid, passages = validate_questions(questions, model, rel, reserved)
                qids[qid] = passages
                qids[qid]['sub'] = dataset[model][qid]['sub']
                qids[qid]['rel'] = dataset[model][qid]['rel']
                qids[qid]['obj'] = dataset[model][qid]['obj']
                reserved.append(qid)
            samples[model] = qids
        with open('./samples.json', 'w') as f:
            json.dump(samples, f, indent=4)
