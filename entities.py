import random
import math
from collections import deque
import logging
from copy import deepcopy
from datetime import datetime, timedelta
import itertools
from statistics import n_teams, matches_expected_time, court_slack_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict

RANDOM_STATE = 40
random.seed(RANDOM_STATE)

logging.basicConfig(
    filename='tournament.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def generate_break_duration(division, stage):
    min_time = court_slack_time.get(division).get(stage)['min']
    avg_time = court_slack_time.get(division).get(stage)['avg']
    # max_time = court_slack_time.get(division).get(stage)['max']

    # Gamma distribution
    shape = 2  # Adjust for skewness
    scale = (avg_time - min_time) / shape  # Scale ensures mean is ~3 after shifting

    break_time = np.random.gamma(shape, scale) + min_time

    return break_time  # No hard cap on max time

class Team:
    def __init__(self, name, start_time):
        self.name = name
        self.is_eliminated = False  # Relevant for double elimination
        self.start_time = start_time
        self.last_match_time = start_time

    def update_time(self, new_time):
        self.last_match_time = new_time

    def eliminate(self):
        self.is_eliminated = True

    def get_last_match_time(self):
        return self.last_match_time

    def __repr__(self):
        return self.name

class Match:
    def __init__(self, team1, team2, division, stage):
        self.team1 = team1
        self.team2 = team2
        # self.match_duration_distribution = None
        self.start_time = None
        self.duration = None  # Will be set during simulation
        self.winner = None
        self.finish_time = None
        self.division = division
        self.stage = stage

    # # todo change to random
    # def generate_match_time(self):
    #     # return matches_expected_time.get(self.division).get(self.stage)['avg']
    #     min_time = matches_expected_time.get(self.division).get(self.stage)['min']
    #     max_time = matches_expected_time.get(self.division).get(self.stage)['max']
    #     return random.uniform(min_time, max_time)

    def generate_match_time(self):
        # log-normal distribution
        min_time = matches_expected_time.get(self.division).get(self.stage)['min']
        avg_time = matches_expected_time.get(self.division).get(self.stage)['avg']
        max_time = matches_expected_time.get(self.division).get(self.stage)['max']

        sigma = 0.6  # Adjust for reasonable spread (can be tuned)
        desired_mean = avg_time - min_time
        mu = np.log(desired_mean) - (sigma ** 2) / 2

        # Generate log-normal samples
        distribution_shift = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]

        # Shift values and truncate below min_time
        match_time = distribution_shift + min_time

        return match_time

    def play(self):
        """Simulate the match duration based on the given distribution."""
        self.duration = self.generate_match_time() # Ensure min time is 10 min
        self.winner = random.choice([self.team1, self.team2])
        return self.duration, self.winner

    def get_winner(self):
        return self.winner

    def __repr__(self):
        return f"{self.team1} vs {self.team2}. Won {self.winner} ({self.duration:.2f} min)"


class Court:
    def __init__(self, court_id, start_time):
        self.court_id = court_id
        self.start_time = start_time
        self.matches = []
        self.current_time = start_time  # Tracks when the court becomes available

    def play_match(self, match):
        self.current_time = match.start_time + timedelta(minutes=match.duration)
        match.finish_time = self.current_time
        self.matches.append(match)

    def account_court_break(self, duration):
        self.current_time += timedelta(minutes=duration)

    def get_current_time(self):
        return self.current_time

    def __repr__(self):
        return f"Court {self.court_id}"


class MatchNodeSingleElimination:
    """Represents a match in the single-elimination tournament tree."""
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent
        self.children = []
        self.player = None

    def is_leaf(self):
        if len(self.children) > 0:
            return False
        return True

    def add_child(self, child_node):
        self.children.append(child_node)

    def n_children(self):
        return len(self.children)

    def assign_team(self, team):
        self.player = team

    def get_parent_node(self):
        return self.parent

    def get_children(self):
        return self.children

    def remove_children(self):
        self.children = []

    def get_player(self):
        return self.player

    def __repr__(self):
        return f"{self.id}"

class TournamentTreeSingleElimination:
    """Generates a single-elimination tournament tree with ordered match queue."""

    def __init__(self, teams):
        """
        :param teams: List of team names.
        """
        self.teams = teams
        self.num_teams = len(teams)

        try:
            assert self.num_teams >= 2
        except AssertionError:
            logging.log('Need at least 2 teams for a tournament')

    def num_leaves(self):
        return len([node.id for node in self.matches_tree_nodes if node.is_leaf()])

    def get_node_by_id(self, id):
        return [node for node in self.matches_tree_nodes if node.id == id][0]

    def remove_node_by_id(self, id):
        return [node for node in self.matches_tree_nodes if node.id != id]

    def print_tree(self):
        for node in self.matches_tree_nodes:
            logging.info('Node %s, children: %s. Team assigned: %s', node, node.get_children(), node.get_player())

    def build_tree(self):
        """Builds a structured elimination bracket based on given teams."""
        self.matches_tree_nodes = [MatchNodeSingleElimination(id=1, parent=None)]
        while self.num_leaves() < self.num_teams:
            node_to_prune_id = min([node.id for node in self.matches_tree_nodes
                                    if node.is_leaf() or node.n_children() == 1])
            node_to_prune = self.get_node_by_id(node_to_prune_id)
            new_leaf = MatchNodeSingleElimination(
                id=len(self.matches_tree_nodes) + 1,
                parent=node_to_prune)
            self.matches_tree_nodes.append(new_leaf)
            node_to_prune.add_child(new_leaf)

        # assign teams to nodes
        team_index = 0
        for node in self.matches_tree_nodes:
            if node.is_leaf():
                node.assign_team(self.teams[team_index])
                team_index += 1
        self.print_tree()


class PlayDivision:
    def __init__(self, name, n_teams, bracket_system, start_time, court_names=['A', 'B', 'C', 'D', 'E', 'F']):
        self.division_name = name
        self.n_teams = n_teams
        self.teams = [Team(str(i), start_time) for i in range(1, self.n_teams + 1)]
        self.bracket_system = bracket_system
        self.games_tree = None
        self.start_time = start_time
        self.matches = []
        self.courts = [Court(name, self.start_time) for name in court_names]
        self.finish_time = None

    def build_division_tree(self):
        if self.bracket_system == 'single_elimination':
            self.games_tree = TournamentTreeSingleElimination(self.teams)
            self.games_tree.build_tree()

    def generate_break_duration(self, stage):
        min_time = court_slack_time.get(self.division_name).get(stage)['min']
        avg_time = court_slack_time.get(self.division_name).get(stage)['avg']
        max_time = court_slack_time.get(self.division_name).get(stage)['max']

        # Gamma distribution
        shape = 2  # Adjust for skewness
        scale = (avg_time - min_time) / shape  # Scale ensures mean is ~3 after shifting

        break_time = np.random.gamma(shape, scale) + min_time

        return break_time  # No hard cap on max time

    def play_matches(self):
        tree_copy = deepcopy(self.games_tree.matches_tree_nodes[::-1])
        court_cycle = itertools.cycle(self.courts)# reverese, starting from the last node
        while len(tree_copy) > 1:
            parent_node = tree_copy[0].get_parent_node()
            player1_node, player2_node = parent_node.get_children()
            player1 = player1_node.get_player()
            player2 = player2_node.get_player()
            stage = 'early_stage' if parent_node.id > 3 else 'semifinal_final'
            current_match = Match(player1, player2,
                                  division=self.division_name, stage=stage)
            current_court = next(court_cycle)
            logging.info('Match on court %s', current_court)
            # play match
            current_match.start_time = max([player1.get_last_match_time(), player2.get_last_match_time(),
                                            current_court.current_time])
            logging.info('Team %s VS Team %s start at %s', player1_node.player, player2_node.player, current_match.start_time)
            _, match_winner = current_match.play()
            current_court.play_match(current_match)

            # update teams last_match_time + some break
            player1_slack = self.generate_break_duration(stage=stage)
            player2_slack = self.generate_break_duration(stage=stage)
            player1.update_time(current_court.get_current_time() + timedelta(minutes=player1_slack))
            player2.update_time(current_court.get_current_time() + timedelta(minutes=player2_slack))
            if player1 == match_winner:
                player2.eliminate()
            else:
                player1.eliminate()

            logging.info('the winner is %s, match duration = %s min. Started at %s, finished at %s',
                         match_winner, current_match.duration, current_match.start_time, current_match.finish_time)
            logging.info('Team %s slack is %s min. Team %s slack is %s min.', player1, player1_slack,
                         player2, player2_slack)

            # break time
            if len(tree_copy) > 1:
                break_duration = self.generate_break_duration(stage=stage)
                current_court.account_court_break(break_duration)
            logging.info('Break on the court %s, finished at %s ', current_court, current_court.get_current_time())


            # remove nodes of matches and reassign
            parent_node.assign_team(current_match.get_winner())
            parent_node.remove_children()
            tree_copy = [node for node in tree_copy if node != player1_node and node != player2_node]
        self.finish_time = current_court.get_current_time()
        logging.info('The last division match took place on the court %s. Finish time: %s', current_court,
                     self.finish_time)

    def get_division_finish_time(self):
        return self.finish_time


class RoundRobinMatch:
    def __init__(self, team1:Team, team2:Team, division, court, previous_matches, match_number):
        self.team1 = team1
        self.team2 = team2
        self.start_time = None
        self.duration = None  # Will be set during simulation
        self.winner = None
        self.finish_time = None
        self.division = division
        self.previous_matches = previous_matches
        self.court = court
        self.match_number = match_number

    def generate_match_time(self):
        # log-normal distribution
        min_time = matches_expected_time.get(self.division).get('early_stage')['min']
        avg_time = matches_expected_time.get(self.division).get('early_stage')['avg']
        max_time = matches_expected_time.get(self.division).get('early_stage')['max']

        sigma = 0.6  # Adjust for reasonable spread (can be tuned)
        desired_mean = avg_time - min_time
        mu = np.log(desired_mean) - (sigma ** 2) / 2

        # Generate log-normal samples
        distribution_shift = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]

        # Shift values and truncate below min_time
        match_time = distribution_shift + min_time

        return match_time

    def play(self):
        if self.previous_matches is None or self.previous_matches == [None]:
            self.start_time = max(self.team1.get_last_match_time(), self.team2.get_last_match_time())
        else:
            self.start_time = max([match.finish_time for match in self.previous_matches])
        self.duration = self.generate_match_time() # Ensure min time is 10 min
        self.winner = random.choice([self.team1, self.team2])
        self.finish_time = self.start_time + timedelta(minutes=self.duration)

        # teams slack
        team1_break = generate_break_duration(self.division, 'early_stage')
        team2_break = generate_break_duration(self.division, 'early_stage')

        # court break
        court_break = generate_break_duration(self.division, 'early_stage')
        self.finish_time = self.start_time + timedelta(minutes=court_break)
        self.team1.update_time(self.finish_time + timedelta(minutes=team1_break))
        self.team2.update_time(self.finish_time + timedelta(minutes=team2_break))
        # return self.duration, self.winner, self.team1.get_last_match_time(), self.team2.get_last_match_time()

    def get_winner(self):
        return self.winner

    def get_match_info(self):
        if self.start_time is None:
            return f"Scheduled match # {self.match_number} between Team {self.team1} and Team {self.team2} on the court {self.court}."
        else:
            return (f"Match {self.match_number} between Team {self.team1} and Team {self.team2} on the court {self.court}. "
                    f"Starts after matches {self.previous_matches}. Match start: {self.start_time}, match finish: {self.finish_time}."
                    f"Team {self.team1} slack finished at "
                    f"{self.team1.get_last_match_time()}. Team {self.team2} slack finished at "
                    f"{self.team2.get_last_match_time()}")

    def get_match_finish_time(self):
        return self.finish_time

    def get_previous_matches(self):
        return self.previous_matches

    def __repr__(self):
        return f"{self.match_number}"


class PlayDivisionRoundRobin:
    def __init__(self, name, n_teams, start_time, bracket_system='round_robin',
                 court_names=['A', 'B', 'C', 'D', 'E', 'F'], team_size_first_priority=3,
                 team_size_second_priority=4):
        self.division_name = name
        self.n_teams = n_teams
        self.teams = [Team(str(i), start_time) for i in range(1, self.n_teams + 1)]
        self.bracket_system = bracket_system
        self.games_tree = None
        self.start_time = start_time
        self.matches = []
        self.groups = []
        self.courts = court_names
        self.finish_time = None
        self.team_size_first_priority = team_size_first_priority
        self.team_size_second_priority = team_size_second_priority
        self.last_court_match = defaultdict(list)

    def build_matches_queue(self):
        # split all teams into groups
        n_teams_first_priority_size = self.n_teams
        while n_teams_first_priority_size % self.team_size_first_priority != 0 \
            and n_teams_first_priority_size >= 0:
            n_teams_first_priority_size -= self.team_size_second_priority
        if n_teams_first_priority_size < 0:
            logging.error("Unable to split the teams into groups of size {} and {}", self.team_size_first_priority,
                          self.team_size_second_priority)
        n_teams_second_priority_size = self.n_teams - n_teams_first_priority_size
        logging.info("%s groups of size %s and %s group of size %s",
                     n_teams_first_priority_size // self.team_size_first_priority,
                     self.team_size_first_priority,
                     n_teams_second_priority_size // self.team_size_second_priority,
                     self.team_size_second_priority)
        court_cycle = itertools.cycle(self.courts)
        for i in range(0, n_teams_first_priority_size, self.team_size_first_priority):
            self.groups.append(self.teams[i: i + self.team_size_first_priority])
        for i in range(n_teams_first_priority_size, self.n_teams, self.team_size_second_priority):
            self.groups.append(self.teams[i: i + self.team_size_second_priority])
        logging.info('Team groups: %s', self.groups)

        # Build matches dependencies
        court_cycle = itertools.cycle(self.courts)
        match_count = 0
        for group in self.groups:

            current_court = next(court_cycle)
            if len(group) == 3:
                match1 = RoundRobinMatch(
                    team1=group[0],
                    team2=group[1],
                    division=self.division_name,
                    court=current_court,
                    previous_matches=[self.last_court_match.get(current_court)],
                    match_number=match_count
                )
                self.matches.append(match1)
                self.last_court_match[current_court] = match1
                match_count += 1

                match2 = RoundRobinMatch(
                    team1=group[0],
                    team2=group[2],
                    division=self.division_name,
                    court=current_court,
                    previous_matches=[self.last_court_match.get(current_court)],
                    match_number=match_count
                )
                self.matches.append(match2)
                self.last_court_match[current_court] = match2
                match_count += 1

                match3 = RoundRobinMatch(
                    team1=group[1],
                    team2=group[2],
                    division=self.division_name,
                    court=current_court,
                    previous_matches=[self.last_court_match.get(current_court)],
                    match_number = match_count
                )
                self.matches.append(match3)
                self.last_court_match[current_court] = match3
                match_count += 1

            if len(group) == 4:
                # skip one court if cannot allocate 2 neighbour courts
                if current_court in ['B', 'D', 'F']:
                    current_court = next(court_cycle)
                court1 = deepcopy(current_court)
                current_court = next(court_cycle)
                court2 = deepcopy(current_court)

                match1 = RoundRobinMatch(
                    team1=group[0],
                    team2=group[1],
                    division=self.division_name,
                    court=court1,
                    previous_matches=[self.last_court_match.get(court1)],
                    match_number=match_count
                )
                self.matches.append(match1)
                self.last_court_match[court1] = match1
                match_count += 1

                match2 = RoundRobinMatch(
                    team1=group[2],
                    team2=group[3],
                    division=self.division_name,
                    court=court2,
                    previous_matches=[self.last_court_match.get(court2)],
                    match_number=match_count
                )
                self.matches.append(match2)
                self.last_court_match[court2] = match2
                match_count += 1

                match3 = RoundRobinMatch(
                    team1=group[0],
                    team2=group[3],
                    division=self.division_name,
                    court=court1,
                    previous_matches=[match1, match2],
                    match_number=match_count
                )
                self.matches.append(match3)
                self.last_court_match[court1] = match3
                match_count += 1

                match4 = RoundRobinMatch(
                    team1=group[2],
                    team2=group[1],
                    division=self.division_name,
                    court=court2,
                    previous_matches=[match1, match2],
                    match_number=match_count
                )
                self.matches.append(match4)
                self.last_court_match[court2] = match4
                match_count += 1

                match5 = RoundRobinMatch(
                    team1=group[0],
                    team2=group[2],
                    division=self.division_name,
                    court=court1,
                    previous_matches=[match3, match4],
                    match_number=match_count
                )
                self.matches.append(match5)
                self.last_court_match[court1] = match5
                match_count += 1

                match6 = RoundRobinMatch(
                    team1=group[3],
                    team2=group[1],
                    division=self.division_name,
                    court=court2,
                    previous_matches=[match3, match4],
                    match_number=match_count
                )
                self.matches.append(match6)
                self.last_court_match[court2] = match6
                match_count += 1
        for match in self.matches:
            logging.info(match.get_match_info())

    def play_matches(self):
        for match in self.matches:
            match.play()
            logging.info(match.get_match_info())
        self.finish_time = max([match.get_match_finish_time() for match in self.matches])
        logging.info(f"Round Robin stage for %s division is finished at %s", self.division_name, self.finish_time)


        # tree_copy = deepcopy(self.games_tree.matches_tree_nodes[::-1])
        # court_cycle = itertools.cycle(self.courts)  # reverese, starting from the last node
        # while len(tree_copy) > 1:
        #     parent_node = tree_copy[0].get_parent_node()
        #     player1_node, player2_node = parent_node.get_children()
        #     player1 = player1_node.get_player()
        #     player2 = player2_node.get_player()
        #     stage = 'early_stage' if parent_node.id > 3 else 'semifinal_final'
        #     current_match = Match(player1, player2,
        #                           division=self.division_name, stage=stage)
        #     current_court = next(court_cycle)
        #     logging.info('Match on court %s', current_court)
        #     # play match
        #     current_match.start_time = max([player1.get_last_match_time(), player2.get_last_match_time(),
        #                                     current_court.current_time])
        #     logging.info('Team %s VS Team %s start at %s', player1_node.player, player2_node.player,
        #                  current_match.start_time)
        #     _, match_winner = current_match.play()
        #     current_court.play_match(current_match)
        #
        #     # update teams last_match_time + some break
        #     player1_slack = self.generate_break_duration(stage=stage)
        #     player2_slack = self.generate_break_duration(stage=stage)
        #     player1.update_time(current_court.get_current_time() + timedelta(minutes=player1_slack))
        #     player2.update_time(current_court.get_current_time() + timedelta(minutes=player2_slack))
        #     if player1 == match_winner:
        #         player2.eliminate()
        #     else:
        #         player1.eliminate()
        #
        #     logging.info('the winner is %s, match duration = %s min. Started at %s, finished at %s',
        #                  match_winner, current_match.duration, current_match.start_time, current_match.finish_time)
        #     logging.info('Team %s slack is %s min. Team %s slack is %s min.', player1, player1_slack,
        #                  player2, player2_slack)
        #
        #     # break time
        #     if len(tree_copy) > 1:
        #         break_duration = self.generate_break_duration(stage=stage)
        #         current_court.account_court_break(break_duration)
        #     logging.info('Break on the court %s, finished at %s ', current_court, current_court.get_current_time())
        #
        #     # remove nodes of matches and reassign
        #     parent_node.assign_team(current_match.get_winner())
        #     parent_node.remove_children()
        #     tree_copy = [node for node in tree_copy if node != player1_node and node != player2_node]
        # self.finish_time = current_court.get_current_time()
        # logging.info('The last division match took place on the court %s. Finish time: %s', current_court,
        #              self.finish_time)

    def get_division_finish_time(self):
        return self.finish_time

    def __repr__(self):
        return f"Tournament finished in {self.total_time:.2f} minutes."

class KendoTournament:
    def __init__(self, u13_n, u16_n, u19_n, women_n, mixed_n, u13_courts, u16_courts, u19_courts, women_courts,
                 mixed_courts, division_start_datetime = datetime(2025,7,13,9)):
        self.tournament_start_datetime = division_start_datetime
        self.u13_n = u13_n
        self.u16_n = u16_n
        self.u19_n = u19_n
        self.women_n = women_n
        self.mixed_n = mixed_n
        self.u13_courts = u13_courts
        self.u16_courts = u16_courts
        self.u19_courts = u19_courts
        self.women_courts = women_courts
        self.mixed_courts = mixed_courts
        self.tournament_finish_datetime = None
        self.u13_finish_time = None
        self.u16_finish_time = None
        self.u19_finish_time = None
        self.women_finish_time = None
        self.mixed_round_robin_finish_time = None

    def play_tournament(self):
        logging.info('-------------------------------PLAY DIVISION U13 SINGLE ELIMINATION----------------------------')
        division_u13 = PlayDivision(
            name='U13',
            n_teams=self.u13_n,
            bracket_system='single_elimination',
            start_time=self.tournament_start_datetime,
            court_names=self.u13_courts
        )
        division_u13.build_division_tree()
        division_u13.play_matches()
        self.u13_finish_time = division_u13.get_division_finish_time()

        logging.info('-------------------------------PLAY DIVISION U16 SINGLE ELIMINATION----------------------------')
        division_u16 = PlayDivision(
            name='U16',
            n_teams=self.u16_n,
            bracket_system='single_elimination',
            start_time=self.tournament_start_datetime,
            court_names=self.u16_courts
        )
        division_u16.build_division_tree()
        division_u16.play_matches()
        self.u16_finish_time = division_u16.get_division_finish_time()

        logging.info('-------------------------------PLAY DIVISION U19 SINGLE ELIMINATION----------------------------')
        division_u19 = PlayDivision(
            name='U19',
            n_teams=self.u19_n,
            bracket_system='single_elimination',
            start_time=self.tournament_start_datetime,
            court_names=self.u19_courts
        )
        division_u19.build_division_tree()
        division_u19.play_matches()
        self.u19_finish_time = division_u19.get_division_finish_time()

        logging.info('-------------------------------PLAY DIVISION WOMEN SINGLE ELIMINATION----------------------------')
        division_women = PlayDivision(
            name='Women',
            n_teams=self.women_n,
            bracket_system='single_elimination',
            start_time=self.tournament_start_datetime,
            court_names=self.women_courts
        )
        division_women.build_division_tree()
        division_women.play_matches()
        self.women_finish_time = division_women.get_division_finish_time()

        # after these divisions finish, Mixed division starts
        mixed_round_robin_start_time = max([division_u13.get_division_finish_time(),
                                         division_u16.get_division_finish_time(),
                                         division_u19.get_division_finish_time(),
                                         division_women.get_division_finish_time()])

        logging.info('-------------------------------PLAY MIXED DIVISION ROUND ROBIN----------------------------')
        division_round_robin_mixed = PlayDivisionRoundRobin(
            name='Mixed',
            n_teams=self.mixed_n,
            court_names=['A', 'B', 'C', 'D', 'E', 'F'],
            team_size_first_priority=3,
            team_size_second_priority=4
        )
        division_round_robin_mixed.build_matches_queue()
        division_round_robin_mixed.play_matches()
        self.mixed_round_robin_finish_time = division_round_robin_mixed.get_division_finish_time()

        # after these divisions finish, Mixed division starts
        mixed_division_start_time = max([division_u13.get_division_finish_time(),
                                         division_u16.get_division_finish_time(),
                                         division_u19.get_division_finish_time(),
                                         division_women.get_division_finish_time()])


        logging.info('-------------------------------PLAY MIXED DIVISION SINGLE ELIMINATION ----------------------------')
        division_mixed = PlayDivision(
            name='Mixed',
            n_teams=self.mixed_n,
            bracket_system='single_elimination',
            start_time=mixed_division_start_time,
            court_names=self.mixed_courts
        )
        division_mixed.build_division_tree()
        division_mixed.play_matches()
        self.tournament_finish_datetime = division_mixed.get_division_finish_time()
        logging.info('Tournament finished at %s', self.tournament_finish_datetime)

    def get_tournament_finish_time(self):
        return self.tournament_finish_datetime

    def get_u13_finish_time(self):
        return self.u13_finish_time

    def get_u16_finish_time(self):
        return self.u16_finish_time

    def get_u19_finish_time(self):
        return self.u19_finish_time

    def get_women_finish_time(self):
        return self.women_finish_time


def build_finish_time_distribution(division_name, timestamps_list, hour_left_boundary, hour_right_boundary):
    timestamps = pd.to_datetime(timestamps_list)

    # Extract the hour of the day in decimal format
    hour_of_day = timestamps.hour + timestamps.minute / 60  # Convert to decimal hours

    # Calculate the 95% quantile (X) in decimal hours
    quantile_95_decimal = np.percentile(hour_of_day, 95)

    # Convert decimal hours to HH:MM format
    quantile_95_hour = int(quantile_95_decimal)  # Extract hours
    quantile_95_minute = int((quantile_95_decimal - quantile_95_hour) * 60)  # Convert fraction to minutes
    quantile_95_str = f"{quantile_95_hour:02d}:{quantile_95_minute:02d}"  # Format as HH:MM

    # Calculate the mean time
    mean_time = timestamps.mean()
    mean_time_str = mean_time.strftime("%H:%M")  # Convert mean time to HH:MM format

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(hour_of_day, bins=24, edgecolor='black', alpha=0.7)

    # Set x-axis limits dynamically
    plt.xlim(hour_left_boundary, hour_right_boundary)

    # Get y-axis max height for proper text box placement
    y_max = max(plt.gca().get_ylim())

    # Add text box with the quantile information (LEFT LOWER CORNER)
    text_quantile = f"{division_name} will finish until {quantile_95_str} with high confidence"
    plt.text(hour_left_boundary + 0.2, y_max * 0.05, text_quantile,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
             fontsize=12)

    # Add text box with the mean time (RIGHT UPPER CORNER)
    text_mean = f"Average finish time: {mean_time_str}"
    plt.text(hour_right_boundary - 1.8, y_max * 0.9, text_mean,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
             fontsize=12)

    plt.xlabel("Finish Time")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {division_name} Simulated Finish Time - Single Elimination")
    plt.xticks(range(hour_left_boundary, hour_right_boundary + 1))  # Show hours dynamically
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

if __name__ == '__main__':

    # finish_time_list = list()
    # u13_finish_time_list = []
    # u16_finish_time_list = []
    # u19_finish_time_list = []
    # women_finish_time_list = []
    # for _ in range(1000):
    #     tournament = KendoTournament(
    #         u13_n=n_teams.get('U13').get('max'),
    #         u16_n=n_teams.get('U16').get('max'),
    #         u19_n=n_teams.get('U19').get('max'),
    #         women_n=n_teams.get('Women').get('max'),
    #         mixed_n=n_teams.get('Mixed').get('max'),
    #         u13_courts=['A'],
    #         u16_courts=['B'],
    #         u19_courts=['C'],
    #         women_courts=['D', 'E', 'F'],
    #         mixed_courts=['A', 'B', 'C', 'D', 'E', 'F']
    #     )
    #     tournament.play_tournament()
    #     u13_finish_time_list.append(tournament.get_u13_finish_time())
    #     u16_finish_time_list.append(tournament.get_u16_finish_time())
    #     u19_finish_time_list.append(tournament.get_u19_finish_time())
    #     women_finish_time_list.append(tournament.get_women_finish_time())
    #     finish_time_list.append(tournament.get_tournament_finish_time())
    #
    # build_finish_time_distribution('The Tournament', finish_time_list, 13, 17)
    # build_finish_time_distribution('Division U13', u13_finish_time_list, 9, 13)
    # build_finish_time_distribution('Division U16', u16_finish_time_list, 9, 13)
    # build_finish_time_distribution('Division U19', u19_finish_time_list, 9, 13)
    # build_finish_time_distribution('Women Division', women_finish_time_list, 9, 13)

    division = PlayDivisionRoundRobin(
        name='Mixed',
        n_teams=22,
        court_names=['A', 'B', 'C', 'D', 'E', 'F'],
        start_time=datetime(2025,7,13,9)
    )
    division.build_matches_queue()
    division.play_matches()


