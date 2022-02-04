import pprint
import random
from collections import defaultdict
import itertools
import time
import pandas as pd
import ast
import pickle
import os
import shutil

def _print():
    print('---------------------------------------------------------------------------------------------------------')

class BoardCache:
    
    def __init__(self, dataset, dataset_path, chunks_path, timer):
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.chunks_path = chunks_path
        self.timer = timer
        self.board_cache = {}

    def build_board_cache_from_chunks(self):
        num_its = 0
        self.timer.start('build_board_cache_from_chunks')
        num_board_chunks_total = len(os.listdir(self.chunks_path))
        for board_chunk_filepath in os.listdir(self.chunks_path)[:10]:
            self.timer.start('board_chunk')
            full_board_chunk_path = self.chunks_path + board_chunk_filepath
            board_chunk_df = pd.read_csv(full_board_chunk_path)
            board_chunk_list = board_chunk_df.to_dict('records')
            for board in board_chunk_list:
                word_guess = board['x']
                word_target = board['y']
                resulting_board = board['z']
                if (word_guess not in self.board_cache):
                    self.board_cache[word_guess] = {}
                if (word_target not in self.board_cache[word_guess]):
                    self.board_cache[word_guess][word_target] = resulting_board
            num_its += 1
            self.timer.end('board_chunk')
            print('time board_chunk {}/{}: {}s'.format(num_its, num_board_chunks_total, self.timer.get('board_chunk')['elapsed']))
        self.timer.end('build_board_cache_from_chunks')
        print('time build_board_cache_from_chunks: {}s'.format(self.timer.get('build_board_cache_from_chunks')['elapsed']))

    def get_board_cache(self):
        return self.board_cache

class Timer:

    def __init__(self):
        self.times = {}

    def start(self, time_label):
        self.times[time_label] = {}
        self.times[time_label]['start'] = time.time()

    def end(self, time_label):
        self.times[time_label]['end'] = time.time()
        self.times[time_label]['elapsed'] = round(self.times[time_label]['end'] - self.times[time_label]['start'],4)

    def get(self, time_label):
        return self.times[time_label]

class WordleLogger:

    def __init__(self, is_verbose):
        self.is_verbose = is_verbose

    def log_target_word(self, word_target):
        if (self.is_verbose):
            _print()
            print('target               : {}'.format(word_target))
            _print()

    def log_guess(self, num_guesses, word_guess, word_guess_payloads):
        if (self.is_verbose):
            top_n = min(len(word_guess_payloads), 5)
            top_n_guesses = ''
            for i in range(top_n):
                top_n_guesses += str((word_guess_payloads[i]['word'], round(word_guess_payloads[i]['word_score'],3)))
            print('guess # {}            : {}'.format(num_guesses, word_guess))
            print('top_n                : {}'.format(top_n_guesses))
            _print()

    def log_solved(self, num_guesses):
        if (self.is_verbose):
            print('solved! total guesses: {}'.format(num_guesses))
            _print()

    def log_run_data(self, num_guesses, time_solve, time_guesses):
        if (self.is_verbose):
            print('num_guesses          : {}'.format(num_guesses))
            print('time_solve           : {}'.format(str(time_solve) + ' s'))
            print('time_guesses         : {}'.format(str(time_guesses)))
            _print()

    def log_multiple_runs_stats(self, trials):
        if (self.is_verbose):
            num_trials = len(trials)
            avg_num_guesses = round(sum([x['num_guesses'] for x in trials])/num_trials,4)
            total_time_solve = round(sum([x['time_solve'] for x in trials]),4)
            avg_time_solve = round(total_time_solve/num_trials,4)
            _print()
            print('number of trials   : {}'.format(num_trials))
            print('avg_num_guesses    : {}'.format(avg_num_guesses))
            print('avg_time_solve     : {}s'.format(avg_time_solve))
            print('total_time_solve   : {}s'.format(total_time_solve))
            _print()

    def log_set_first_word_guess(self, time_elapsed):
        if (self.is_verbose):
            _print()
            print('set_first_word_guess time: {}s'.format(time_elapsed))
            _print()

    def log_best_guess_given_boards(self, word_guess):
        if (self.is_verbose):
            _print()
            print('best_guess_given_boards: {}'.format(word_guess))
            _print()


    def log_multiple_trials_completion_status(self, trial_idx, num_trials_total):
        print('completed trial {} out of {}'.format(trial_idx, num_trials_total))


class WordleSolver:
    def __init__(self, dataset, is_verbose):
        self.dataset = dataset
        self.dataset_path = 'data/{}/words.txt'.format(dataset)
        self.chunks_path = 'data/{}/chunks/'.format(dataset)
        self.first_word_guess_path = 'data/{}/first_guess.txt'.format(dataset)
        self.is_verbose = is_verbose
        self.timer = Timer()
        self.wordle_logger = WordleLogger(is_verbose)

    def upload_chunks(self):
        chunk_id = 1
        with open(self.dataset_path) as f:
            words_list = f.readlines()
            words_list = [word.strip() for word in words_list]
        rows = []
        for word_1 in words_list:
            for word_2 in words_list:
                board = self._get_board(word_1, word_2, False)
                row = {}
                row['x'] = word_1
                row['y'] = word_2
                row['z'] = board
                rows.append(row)
                if (len(rows) == 1000000):
                    rows_df = pd.DataFrame(rows)
                    rows_df.to_csv('{}{}.csv'.format(self.chunks_path, chunk_id), index=False)
                    rows = []
                    chunk_id += 1
        rows_df = pd.DataFrame(rows)
        rows_df.to_csv('{}{}.csv'.format(self.chunks_path, chunk_id), index=False)

    def set_first_word_guess(self):
        self.timer.start('set_first_word_guess')
        with open('data/{}/words.txt'.format(self.dataset)) as f:
            words_list = f.readlines()
            words_list = [word.strip() for word in words_list]
        word_guess_payloads = []
        for possible_word in words_list:
            word_score = self._get_word_score(possible_word, words_list)
            word_guess_payload = {
                'word': possible_word,
                'word_score': word_score,
            }
            word_guess_payloads.append(word_guess_payload)
        word_guess_payloads.sort(key = lambda x: x['word_score'], reverse=True)
        first_word_guess = word_guess_payloads[0]['word']

        word_guess_payloads_list_form = ''
        for payload in word_guess_payloads:
            word_guess_payloads_list_form += str(payload) + '\n'

        self.timer.end('set_first_word_guess')
        time_elapsed = self.timer.get('set_first_word_guess')['elapsed']
        self.wordle_logger.log_set_first_word_guess(time_elapsed)
        with open(self.first_word_guess_path, 'w') as f:
            f.write('{}\n{}\n{}'.format(first_word_guess, word_guess_payloads, word_guess_payloads_list_form))

    def get_first_word_guess(self):
        with open(self.first_word_guess_path) as f:
            first_word_guess = f.readlines()[0]
            first_word_guess = first_word_guess.strip()
            return first_word_guess

    def init_board_cache(self):
        board_cache = BoardCache(self.dataset, self.dataset_path, self.chunks_path, self.timer)
        board_cache.build_board_cache_from_chunks()
        self.board_cache = board_cache.get_board_cache()

    def _get_word_score(self, word, possible_words):
        word_score = 0
        num_possible_words = len(possible_words)

        board_result_combinations = itertools.product(['_', 'G', 'Y'], repeat=5)
        for board_result in board_result_combinations:
            board = ''
            for i,br in enumerate(list(board_result)):
                board += word[i] + br
            expected_board_probability = self._get_expected_board_probability(board, word, possible_words)
            num_possible_words_given_board = len(self._get_possible_words_given_board(board, possible_words))
            expected_num_words_eliminated_with_board = num_possible_words - num_possible_words_given_board
            print('exp_bp', expected_board_probability)
            print('exp_nwe', num_possible_words_given_board)
            word_score += expected_board_probability * expected_num_words_eliminated_with_board
        return word_score

    def _get_board(self, word_guess, word_target, use_cache = True):
        if (use_cache):
            if (word_guess in self.board_cache and word_target in self.board_cache[word_guess]):
                return self.board_cache[word_guess][word_target]
        if (word_guess == '*****'):
            return '*_*_*_*_*_'
        
        word_guess_list = list(word_guess)
        word_target_list = list(word_target)
        feedback = [None for i in range(5)]
        letters_in_word_target = defaultdict(int)
        letters_in_word_guess = defaultdict(int)
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        for letter in letters:
            letters_in_word_target[letter] = 0
            letters_in_word_guess[letter] = 0
        for c in word_target_list:
            letters_in_word_target[c] += 1
        for c in word_guess_list:
            letters_in_word_guess[c] += 1

        # G
        for i in range(len(word_guess)):
            c_guess = word_guess_list[i]
            c_target = word_target_list[i]
            if (c_guess == c_target):
                feedback[i] = c_guess + 'G'
                word_guess_list[i] = '-'
                word_target_list[i] = '-'
                letters_in_word_guess[c_guess] -= 1
                letters_in_word_target[c_guess] -= 1
        # Y or _
        for i,c in enumerate(word_guess_list):
            if (c != '-'):
                if letters_in_word_target[c] > 0 and letters_in_word_guess[c] > 0:
                    feedback[i] = c + 'Y'
                    letters_in_word_target[c]-=1
                    letters_in_word_guess[c]-=1
                else:
                    feedback[i] = c + '_'
        board = ''
        for position in feedback:
            board += position
        return board


    def _get_expected_board_probability(self, board, word, possible_words):
        num_possible_words = len(possible_words)
        num_words_possible_given_word_and_board = 0
        for possible_word in possible_words:
            is_possible_given_word_and_board = True
            expected_board = self._get_board(word, possible_word)
            is_possible_given_word_and_board = board == expected_board
            if (is_possible_given_word_and_board):
                num_words_possible_given_word_and_board += 1
        expected_board_probability = num_words_possible_given_word_and_board/num_possible_words
        return expected_board_probability
        
    def _get_possible_words_given_board(self, board, words_list):
        possible_words = []
        word_guess = ''
        for i in range(0, len(board), 2):
            word_guess += board[i]
        for word_target in words_list:
            expected_board = self._get_board(word_guess, word_target)
            is_word_target_possible = board == expected_board
            if (is_word_target_possible):
                possible_words.append(word_target)
        return possible_words

    def _get_word_score(self, word, possible_words):
        word_score = 0
        num_possible_words = len(possible_words)

        board_result_combinations = itertools.product(['_', 'G', 'Y'], repeat=5)
        for board_result in board_result_combinations:
            board = ''
            for i,br in enumerate(list(board_result)):
                board += word[i] + br
            expected_board_probability = self._get_expected_board_probability(board, word, possible_words)
            num_possible_words_given_board = len(self._get_possible_words_given_board(board, possible_words))
            expected_num_words_eliminated_with_board = num_possible_words - num_possible_words_given_board
            word_score += expected_board_probability * expected_num_words_eliminated_with_board
        return word_score

    def solve_single_run(self, defined_word_target = ''):
        self.timer.start('solve')

        # get list of 5-letter words: https://www.bestwordlist.com/5letterwords.htm
        with open('data/{}/words.txt'.format(self.dataset)) as f:
            words_list = f.readlines()
            words_list = [word.strip() for word in words_list]
        # words_list = list(random.sample(words_list, 100))

        # start guessing!
        num_guesses = 0
        word_target = defined_word_target if defined_word_target != '' else random.sample(words_list,1)[0]
        word_guess = None

        boards = ['*_*_*_*_*_']
        self.wordle_logger.log_target_word(word_target)
        time_guesses = []
        possible_words = words_list
        while word_guess != word_target and num_guesses < 10:
            self.timer.start('guess')
            num_guesses += 1
            possible_words = self._get_possible_words_given_board(boards[-1], possible_words)

            # treat first guess as special, because there is always the same optimal first guess for a given dataset
            word_guess_payloads = []
            if num_guesses == 1:
                word_guess = self.get_first_word_guess()
            else:
                for possible_word in possible_words:
                    word_score = self._get_word_score(possible_word, possible_words)
                    word_guess_payload = {
                        'word': possible_word,
                        'word_score': word_score,
                    }
                    word_guess_payloads.append(word_guess_payload)
                word_guess_payloads.sort(key = lambda x: x['word_score'], reverse=True)
                word_guess = word_guess_payloads[0]['word']
            curr_board = self._get_board(word_guess, word_target)
            boards.append(curr_board)
            self.wordle_logger.log_guess(num_guesses, word_guess, word_guess_payloads)

            self.timer.end('guess')
            time_guesses.append(self.timer.get('guess')['elapsed'])
        
        self.wordle_logger.log_solved(num_guesses)
        self.timer.end('solve')
        time_solve = self.timer.get('solve')['elapsed']
        self.wordle_logger.log_run_data(num_guesses, time_solve, time_guesses)
        return num_guesses, time_solve, time_guesses


    def solve_multiple_runs(self, num_runs):
        trials = []


        trial_idx = 0
        for i in range(num_runs):
            num_guesses, time_solve, time_guesses = self.solve_single_run()
            trial = {
                'num_guesses': num_guesses,
                'time_solve': time_solve,
                'time_guesses': time_guesses,
            }
            trials.append(trial)

            trial_idx += 1
            if (trial_idx % 10 == 0):
                self.wordle_logger.log_multiple_trials_completion_status(trial_idx, num_runs)
        
        self.wordle_logger.log_multiple_runs_stats(trials)

    def solve_against_dataset(self):
        with open('data/{}/words.txt'.format(self.dataset)) as f:
            words_list = f.readlines()
            words_list = [word.strip() for word in words_list]

        trials = []
        trial_idx = 0
        for word in words_list:
            num_guesses, time_solve, time_guesses = self.solve_single_run(word)
            trial = {
                'num_guesses': num_guesses,
                'time_solve': time_solve,
                'time_guesses': time_guesses,
            }
            trials.append(trial)

            trial_idx += 1
            if (trial_idx % 10 == 0):
                self.wordle_logger.log_multiple_trials_completion_status(trial_idx, len(words_list))
        
        self.wordle_logger.log_multiple_runs_stats(trials)

    def get_best_guess_given_boards(self, boards):
        num_boards = len(boards)
        if (num_boards == 1):
            word_guess = self.get_first_word_guess()
            self.wordle_logger.log_best_guess_given_boards(word_guess)
            return word_guess

        with open('data/{}/words.txt'.format(self.dataset)) as f:
            words_list = f.readlines()
            words_list = [word.strip() for word in words_list]

        possible_words = words_list
        for i in range(1, len(boards)):
            possible_words = self._get_possible_words_given_board(boards[i], possible_words)

        word_guess_payloads = []
        for possible_word in possible_words:
            word_score = self._get_word_score(possible_word, possible_words)
            word_guess_payload = {
                'word': possible_word,
                'word_score': word_score,
            }
            word_guess_payloads.append(word_guess_payload)
        word_guess_payloads.sort(key = lambda x: x['word_score'], reverse=True)
        word_guess = word_guess_payloads[0]['word']
        self.wordle_logger.log_best_guess_given_boards(word_guess)
        self.wordle_logger.log_guess(num_boards, word_guess, word_guess_payloads)
        return word_guess


ws = WordleSolver('c', True)
# ws.upload_chunks() # run once to store chunks of pre-calculated boards
ws.init_board_cache()
# ws.set_first_word_guess() # run once to get first_word_guess
# ws.solve_single_run()
# ws.solve_multiple_runs(10)
ws.solve_against_dataset()




# ## test against online Wordle
# ## 2022_02_03
# wordle_boards = [
#     '*_*_*_*_*_',
#     'RYAYI_SYE_',
#     'SGT_RYAYP_',
#     # '*_*_*_*_*_',
#     # '*_*_*_*_*_',
# ]
# ws.get_best_guess_given_boards(wordle_boards)