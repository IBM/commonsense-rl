import argparse
import numpy as np
import random
from time import time
import torch
import pickle
import agent
from utils import extractor
from utils.generic import getUniqueFileHandler
from utils.kg import construct_kg, load_manual_graphs, RelationExtractor
from utils.textworld_utils import get_goal_graph
from utils.nlp import Tokenizer
from games import dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(agent, opt, random_action=False):
    filter_examine_cmd = False
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    game_path = opt.game_dir + "/" + (
        str(opt.difficulty_level) + "/" + opt.mode  if opt.difficulty_level != '' else opt.game_dir + "/" + opt.mode )
    manual_world_graphs = {}
    if opt.graph_emb_type and 'world' in opt.graph_type:
        print("Loading Knowledge Graph ... ", end='')
        agent.kg_graph, _, _= construct_kg(game_path + '/conceptnet_subgraph.txt')
        print(' DONE')
        # optional: Use complete or brief manually extracted conceptnet subgraph for the agent
        print("Loading Manual World Graphs ... ", end='')
        manual_world_graphs = load_manual_graphs(game_path + '/manual_subgraph_brief')

    if opt.game_name:
        game_path = game_path + "/"+ opt.game_name

    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, opt.max_step_per_episode, opt.batch_size,
                                                opt.mode, opt.verbose)
    # Get Goals as graphs
    goal_graphs = {}
    for game_file in env.gamefiles:
        goal_graph = get_goal_graph(game_file)
        if goal_graph:
            game_id = game_file.split('-')[-1].split('.')[0]
            goal_graphs[game_id] = goal_graph

    # Collect some statistics: nb_steps, final reward.
    total_games_count = len(game_file_names)
    game_identifiers, avg_moves, avg_scores, avg_norm_scores, max_poss_scores = [], [], [], [], []

    for no_episode in (range(opt.nepisodes)):
        if not random_action:
            seed = opt.seed + no_episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            env.seed(seed)

        agent.start_episode(opt.batch_size)
        avg_eps_moves, avg_eps_scores, avg_eps_norm_scores = [], [], []
        num_games = total_games_count
        game_max_scores = []
        game_names = []
        while num_games > 0:
            obs, infos = env.reset()  # Start new episode.
            if filter_examine_cmd:
                for commands_ in infos["admissible_commands"]: # [open refri, take apple from refrigeration]
                    for cmd_ in [cmd for cmd in commands_ if cmd.split()[0] in ["examine", "look"]]:
                        commands_.remove(cmd_)

            batch_size = len(obs)
            num_games -= len(obs)
            game_goal_graphs = [None] * batch_size
            max_scores = []
            game_ids = []
            game_manual_world_graph = [None] * batch_size
            for b, game in enumerate(infos["game"]):
                max_scores.append(game.max_score)
                if "uuid" in game.metadata:
                    game_id = game.metadata["uuid"].split("-")[-1]
                    game_ids.append(game_id)
                    game_names.append(game_id)
                    game_max_scores.append(game.max_score)
                    if len(goal_graphs):
                        game_goal_graphs[b] = goal_graphs[game_id]
                    if len(manual_world_graphs):
                        game_manual_world_graph[b] = manual_world_graphs[game_id]

            if not game_ids:
                game_ids = range(num_games,num_games+batch_size)
                game_names.extend(game_ids)

            commands = ["restart"]*len(obs)
            scored_commands = [[] for b in range(batch_size)]
            last_scores = [0.0]*len(obs)
            scores = [0.0]*len(obs)
            dones = [False]*len(obs)
            nb_moves = [0]*len(obs)
            infos["goal_graph"] = game_goal_graphs
            infos["manual_world_graph"] = game_manual_world_graph
            agent.reset_parameters(opt.batch_size)
            for step_no in range(opt.max_step_per_episode):
                nb_moves = [step + int(not done) for step, done in zip(nb_moves, dones)]

                if agent.graph_emb_type and ('local' in agent.graph_type or 'world' in agent.graph_type):
                    agent.update_current_graph(obs, commands, scored_commands, infos, opt.graph_mode)

                commands = agent.act(obs, scores, dones, infos, scored_commands, random_action)
                obs, scores, dones, infos = env.step(commands)
                infos["goal_graph"] = game_goal_graphs
                infos["manual_world_graph"] = game_manual_world_graph

                for b in range(batch_size):
                    if scores[b] - last_scores[b] > 0:
                        last_scores[b] = scores[b]
                        scored_commands[b].append(commands[b])

                if all(dones):
                    break
                if step_no == opt.max_step_per_episode - 1:
                    dones = [True for _ in dones]
            agent.act(obs, scores, dones, infos, scored_commands, random_action)  # Let the agent know the game is done.

            if opt.verbose:
                print(".", end="")
            avg_eps_moves.extend(nb_moves)
            avg_eps_scores.extend(scores)
            avg_eps_norm_scores.extend([score/max_score for score, max_score in zip(scores, max_scores)])
        if opt.verbose:
            print("*", end="")
        agent.end_episode()
        game_identifiers.append(game_names)
        avg_moves.append(avg_eps_moves) # episode x # games
        avg_scores.append(avg_eps_scores)
        avg_norm_scores.append(avg_eps_norm_scores)
        max_poss_scores.append(game_max_scores)
    env.close()
    game_identifiers = np.array(game_identifiers)
    avg_moves = np.array(avg_moves)
    avg_scores = np.array(avg_scores)
    avg_norm_scores = np.array(avg_norm_scores)
    max_poss_scores = np.array(max_poss_scores)
    if opt.verbose:
        idx = np.apply_along_axis(np.argsort, axis=1, arr=game_identifiers)
        game_avg_moves = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_moves))), axis=0)
        game_norm_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_norm_scores))), axis=0)
        game_avg_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_scores))), axis=0)

        msg = "\nGame Stats:\n-----------\n" + "\n".join(
            "  Game_#{} = Score: {:5.2f} Norm_Score: {:5.2f} Moves: {:5.2f}/{}".format(game_no,avg_score,
                                                                                            norm_score, avg_move,
                                                                                            opt.max_step_per_episode)
            for game_no, (norm_score, avg_score, avg_move) in
            enumerate(zip(game_norm_scores, game_avg_scores, game_avg_moves)))

        print(msg)

        total_avg_moves = np.mean(game_avg_moves)
        total_avg_scores = np.mean(game_avg_scores)
        total_norm_scores = np.mean(game_norm_scores)
        msg = opt.mode+" stats: avg. score: {:4.2f}; norm. avg. score: {:4.2f}; avg. steps: {:5.2f}; \n"
        print(msg.format(total_avg_scores, total_norm_scores,total_avg_moves))

        ## Dump log files ......
        str_result = {opt.mode + 'game_ids': game_identifiers, opt.mode + 'max_scores': max_poss_scores,
                      opt.mode + 'scores_runs': avg_scores, opt.mode + 'norm_score_runs': avg_norm_scores,
                      opt.mode + 'moves_runs': avg_moves}

        results_ofile = getUniqueFileHandler(opt.results_filename + '_' +opt.mode+'_results')
        pickle.dump(str_result, results_ofile)
    return avg_scores, avg_norm_scores, avg_moves


if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser(add_help=False)

    # game files and other directories
    parser.add_argument('--game_dir', default='./games/twc', help='Location of the game e.g ./games/testbed')
    parser.add_argument('--game_name', help='Name of the game file e.g., kitchen_cleanup_10quest_1.ulx, *.ulx, *.z8')
    parser.add_argument('--results_dir', default='./results', help='Path to the results files')
    parser.add_argument('--logs_dir', default='./logs', help='Path to the logs files')
    parser.add_argument('--pretrained_model', required=True, help='Location of the pretrained command scorer model')

    # optional arguments (if game_name is given) for game files
    parser.add_argument('--batch_size', type=int, default='1', help='Number of the games per batch')
    parser.add_argument('--difficulty_level', default='easy', choices=['easy','medium', 'hard'],
                        help='difficulty level of the games')

    # Experiments
    parser.add_argument('--initial_seed', type=int, default=42)
    parser.add_argument('--nruns', type=int, default=5)
    parser.add_argument('--runid', type=int, default=0)
    parser.add_argument('--no_train_episodes', type=int, default=100)
    parser.add_argument('--no_eval_episodes', type=int, default=5)
    parser.add_argument('--train_max_step_per_episode', type=int, default=50)
    parser.add_argument('--eval_max_step_per_episode', type=int, default=50)
    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--hidden_size', type=int, default=300, help='num of hidden units for embeddings')
    parser.add_argument('--hist_scmds_size', type=int, default=3,
                help='Number of recent scored command history to use. Useful when the game has intermediate reward.')
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--token_extractor', default='max', help='token extractor: (any or max)')
    parser.add_argument('--corenlp_url', default='http://localhost:9000/',
                        help='URL for Stanford CoreNLP OpenIE Server for the relation extraction for the local graph')

    parser.add_argument('--noun_only_tokens', action='store_true', default=False,
                        help=' Allow only noun for the token extractor')
    parser.add_argument('--use_stopword', action='store_true', default=False,
                        help=' Use stopwords for the token extractor')
    parser.add_argument('--agent_type', default='knowledgeaware', choices=['random','simple', 'knowledgeaware'],
                        help='Agent type for the text world: (random, simple, knowledgeable)')
    parser.add_argument('--graph_type', default='', choices=['', 'local','world'],
                        help='What type of graphs to be generated')
    parser.add_argument('--graph_mode', default='evolve', choices=['full', 'evolve'],
                        help='Give Full ground truth graph or evolving knowledge graph: (full, evolve)')
    parser.add_argument('--split', default='test', choices=['test', 'valid'],
                        help='Whether to run on test or valid')

    parser.add_argument('--local_evolve_type', default='direct', choices=['direct', 'ground'],
                        help='Type of the generated/evolving strategy for local graph')
    parser.add_argument('--world_evolve_type', default='cdc',
                        choices=['DC','CDC', 'NG','NG+prune','manual'],
                        help='Type of the generated/evolving strategy for world graph')

    # Embeddings
    parser.add_argument('--emb_loc', default='embeddings/', help='Path to the embedding location')
    parser.add_argument('--word_emb_type', default='glove',
                        help='Embedding type for the observation and the actions: ...'
                             '(random, glove, numberbatch, fasttext). Use utils.generic.load_embedings ...'
                             ' to take car of the custom embedding locations')
    parser.add_argument('--graph_emb_type', help='Knowledge Graph Embedding type for actions: (numberbatch, complex)')
    parser.add_argument('--egreedy_epsilon', type=float, default=0.0, help="Epsilon for the e-greedy exploration")

    opt = parser.parse_args()
    print(opt)
    random.seed(opt.initial_seed)
    np.random.seed(opt.initial_seed)
    torch.manual_seed(opt.initial_seed)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.initial_seed)
        torch.backends.cudnn.deterministic = True
    # yappi.start()

    scores_runs = []
    norm_score_runs = []
    moves_runs = []
    test_scores_runs = []
    test_norm_score_runs = []
    test_moves_runs = []

    random_action = False
    if opt.agent_type == 'random':
        random_action = True
        opt.graph_emb_type = None
    if opt.agent_type == 'simple':
        opt.graph_type = ''
        opt.graph_emb_type = None


    tk_extractor = extractor.get_extractor(opt.token_extractor)

    results_filename = opt.results_dir + '/' + opt.agent_type + '_' + opt.game_dir.split('/')[-1] + '_' + (
        opt.graph_mode + '_' + opt.graph_type + '_' if opt.graph_type else '') + (
                           str(opt.word_emb_type) + '_' if opt.word_emb_type else '') + (
                           str(opt.graph_emb_type) + '-' if opt.graph_emb_type else '') + str(
        opt.nruns) + 'runs_' + str(opt.no_train_episodes) + 'episodes_' + str(opt.hist_scmds_size) + 'hsize_' + str(
        opt.egreedy_epsilon) + 'eps_' + opt.difficulty_level+'_' +  opt.local_evolve_type+'_' +  opt.world_evolve_type + '_' + str(opt.runid) + 'runId'
    opt.results_filename = results_filename
    graph = None
    seeds = [random.randint(1, 100) for _ in range(opt.nruns)]
    for n in range(0, opt.nruns):
        opt.run_no = n
        opt.seed = seeds[n]
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)  # For reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opt.seed)

        print("Testing ...")
        emb_types = [opt.word_emb_type, opt.graph_emb_type]
        tokenizer = Tokenizer(noun_only_tokens=opt.noun_only_tokens, use_stopword=opt.use_stopword, ngram=opt.ngram,
                              extractor=tk_extractor)
        rel_extractor = RelationExtractor(tokenizer, openie_url=opt.corenlp_url)
        myagent = agent.KnowledgeAwareAgent(graph, opt, tokenizer,rel_extractor, device)
        myagent.type = opt.agent_type

        print('Loading Pretrained Model ...',end='')
        myagent.model.load_state_dict(torch.load(opt.pretrained_model))
        print('DONE')

        myagent.test(opt.batch_size)
        opt.mode = opt.split
        opt.nepisodes=opt.no_eval_episodes # for testing
        opt.max_step_per_episode = opt.eval_max_step_per_episode
        starttime = time()
        print("\n RUN ", n, "\n")
        test_scores, test_norm_scores, test_moves = play(myagent, opt, random_action=random_action)
        print("Tested in {:.2f} secs".format(time() - starttime))



